// Standalone mechanism only. Does not load model weights or derive learned QSA
// scores. Compact arrays exist only as independent correctness/timing oracles.
#include "paged_sdpa_reader.h"
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>
#include <mlx/version.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;
namespace pg = anemlx::paged;
bool asynchronous = false;
int checks = 0, cases = 0;
void require(bool ok, const std::string& message) {
  ++checks;
  if (!ok) throw std::runtime_error(message);
}
mx::Stream gpu() { return mx::default_stream(mx::Device(mx::Device::gpu)); }
void ready(mx::array& a, mx::array& b) {
  if (asynchronous) { mx::async_eval(a, b); mx::synchronize(gpu()); }
  else mx::eval(a, b);
  a.wait(); b.wait();
}
std::vector<uint16_t> bits(const mx::array& a) {
  require(a.is_available() && a.dtype() == mx::bfloat16 && a.ndim() == 4,
          "Expected evaluated rank-four BF16 input");
  std::vector<uint16_t> out; out.reserve(a.size());
  const auto* p = a.data<uint16_t>();
  for (int i = 0; i < a.shape(0); ++i)
    for (int h = 0; h < a.shape(1); ++h)
      for (int t = 0; t < a.shape(2); ++t)
        for (int d = 0; d < a.shape(3); ++d)
          out.push_back(p[i*a.strides(0) + h*a.strides(1) + t*a.strides(2) + d*a.strides(3)]);
  return out;
}
void equal_bits(const mx::array& a, const mx::array& b, const std::string& label) {
  require(a.shape() == b.shape(), label + ": shape mismatch");
  const auto lhs = bits(a), rhs = bits(b);
  const auto mismatch = std::mismatch(lhs.begin(), lhs.end(), rhs.begin());
  require(mismatch.first == lhs.end(), label + ": BF16 mismatch at " +
          std::to_string(std::distance(lhs.begin(), mismatch.first)));
  for (uint16_t x : lhs)
    require(std::isfinite(std::bit_cast<float>(uint32_t(x) << 16)), label + ": nonfinite result");
}
float sample(int h, int t, int d, bool value) {
  const int x = (h*71 + t*43 + d*19 + (value ? 137 : 11)) % 509 - 254;
  return float(x) / (value ? 16.0f : 128.0f);
}
mx::array head_major(int heads, int capacity, int logical, bool value) {
  std::vector<float> data(size_t(heads) * capacity * 256);
  for (int h = 0; h < heads; ++h)
    for (int t = 0; t < capacity; ++t)
      for (int d = 0; d < 256; ++d)
        data[(size_t(h)*capacity+t)*256+d] = t < logical ? sample(h,t,d,value)
                                                                      : (value ? -80.0f-h : 48.0f+h);
  return mx::array(data.begin(), {1, heads, capacity, 256}, mx::bfloat16);
}
std::vector<int32_t> identity_ids(int n) {
  std::vector<int32_t> ids((n+31)/32); std::iota(ids.begin(), ids.end(), 0); return ids;
}
mx::array page_major(const std::vector<int32_t>& ids, int physical, int n, bool value) {
  std::vector<float> data(size_t(physical)*2*32*256, value ? -80.0f : 48.0f);
  for (int h = 0; h < 2; ++h)
    for (int t = 0; t < n; ++t)
      for (int d = 0; d < 256; ++d)
        data[((size_t(ids[t/32])*2+h)*32+t%32)*256+d] = sample(h,t,d,value);
  return mx::array(data.begin(), {physical,2,32,256}, mx::bfloat16);
}
struct Fixture {
  std::vector<int32_t> ids;
  int physical, capacity;
  pg::StorageKind kind;
  mx::array k, v;
};
Fixture fixture(int n, int layout) {
  auto ids = identity_ids(n);
  if (layout <= 1) {
    const int physical = int(ids.size()) + 5;
    if (layout == 1) {
      int stride = 17; while (std::gcd(stride, physical) != 1) ++stride;
      for (int i = 0; i < int(ids.size()); ++i) ids[i] = (i*stride+3) % physical;
      auto unique = ids; std::sort(unique.begin(), unique.end());
      require(std::adjacent_find(unique.begin(),unique.end()) == unique.end(), "Fixture physical pages alias");
    }
    return {ids,physical,physical*32,pg::StorageKind::PageMajor,
            page_major(ids,physical,n,false),page_major(ids,physical,n,true)};
  }
  const int capacity = layout == 2 ? ((n+1+255)/256)*256 : n+5;
  // V deliberately has a different capacity/head stride from K.
  auto kb = head_major(2,capacity,n,false), vb = head_major(2,capacity+7,n,true);
  auto k = mx::slice(kb,{0,0,0,0},{1,2,n,256},gpu());
  auto v = mx::slice(vb,{0,0,0,0},{1,2,n,256},gpu());
  return {ids,int(ids.size()),capacity,pg::StorageKind::HeadMajor,std::move(k),std::move(v)};
}
struct Mask { std::string name; std::vector<uint8_t> visible; };
std::vector<Mask> masks(int n) {
  std::vector<Mask> result{{"none",{}},{"all_true",std::vector<uint8_t>(n,1)},
                           {"all_false",std::vector<uint8_t>(n,0)},
                           {"last_only",std::vector<uint8_t>(n,0)}};
  result.back().visible.back() = 1;
  if (n > 2051) {
    Mask qsa{"representative_qsa_512x4_plus_tail",std::vector<uint8_t>(n,0)};
    const int count = n/4; int stride = 73;
    while (std::gcd(stride,count) != 1) ++stride;
    for (int i = 0; i < 512; ++i) {
      int block = (i*stride+17)%count;
      for (int lane=0;lane<4;++lane) qsa.visible[block*4+lane]=1;
    }
    for (int t=count*4;t<n;++t) qsa.visible[t]=1;
    require(std::accumulate(qsa.visible.begin(),qsa.visible.end(),0)==2048+n%4,"QSA fixture count mismatch");
    result.push_back(std::move(qsa));
  }
  return result;
}
std::optional<mx::array> mask_array(const Mask& m, int n) {
  if (m.visible.empty()) return std::nullopt;
  if (m.name=="all_true") {
    auto one=mx::array(m.visible.begin(), {1,1,1,1}, mx::bool_);
    return mx::broadcast_to(one,{1,1,1,n},gpu());  // legal zero-stride mask
  }
  return mx::array(m.visible.begin(), {1,1,1,n}, mx::bool_);
}
mx::array reference(const mx::array& q, const mx::array& k, const mx::array& v,
                    const std::optional<mx::array>& mask) {
  return mx::fast::scaled_dot_product_attention(q,k,v,1.0f/16.0f,
      mask ? "array" : "",mask,{},false,gpu());
}
void length_case(int n, const pg::IdentityPageTable& identity) {
  static const char* labels[] = {"page_major_identity_with_holes","page_major_permuted_with_holes",
                                "head_major_capacity256","head_major_clamped_non32"};
  auto q = head_major(24,1,1,false);
  auto compact_k = head_major(2,n,n,false), compact_v = head_major(2,n,n,true);
  ready(compact_k,compact_v);
  for (int layout=0;layout<4;++layout) {
    auto f = fixture(n,layout); ready(f.k,f.v);
    pg::PageTable table(f.ids,f.physical,n);
    const auto old_k=bits(f.k),old_v=bits(f.v);
    const auto k_identity=f.k.buffer().ptr(), v_identity=f.v.buffer().ptr();
    if (layout>=2) {
      require(f.k.strides(1)==int64_t(f.capacity)*256,"Capacity head stride changed");
      require(f.v.strides(1)==int64_t(f.capacity+7)*256,"Independent V head stride changed");
      equal_bits(f.k,compact_k,"Identity K inputs");equal_bits(f.v,compact_v,"Identity V inputs");
    }
    for (const auto& item:masks(n)) {
      auto mask=mask_array(item,n); const auto before=pg::encoded_reads();
      auto actual=layout>=2 ? pg::read_identity(q,f.k,f.v,identity,mask,gpu())
                           : pg::read(q,f.k,f.v,table,f.kind,mask,gpu());
      require(pg::encoded_reads()==before,"Constructing a lazy graph unexpectedly encoded it");
      auto expected=reference(q,compact_k,compact_v,mask); ready(actual,expected);
      require(pg::encoded_reads()==before+1,"Missing/duplicate native read encoding");
      equal_bits(actual,expected,"Paged reader versus compact stock SDPA");
      const auto output=bits(actual);
      if (item.name=="all_false")
        require(std::all_of(output.begin(),output.end(),[](uint16_t x){return x==0;}),"Fully masked output is not exact zero");
      if (item.name=="last_only") {
        const auto vv=bits(compact_v);
        for (int h=0;h<24;++h)
          for (int d=0;d<256;++d)
            require(output[h*256+d]==vv[(size_t(h/12)*n+n-1)*256+d],"Wrong logical tail or GQA head mapping");
      }
      const auto plan=pg::dispatch_info(n,gpu());++cases;
      std::cout<<"{\"event\":\"reader_case\",\"passed\":true,\"tokens\":"<<n
               <<",\"layout\":"<<std::quoted(labels[layout])<<",\"mask\":"<<std::quoted(item.name)
               <<",\"page_tokens\":32,\"physical_pages\":"<<f.physical<<",\"partial_tail\":"<<n%32
               <<",\"two_pass\":"<<plan.two_pass<<",\"blocks\":"<<plan.blocks
               <<",\"logical_scratch_bytes\":"<<plan.logical_scratch_bytes
               <<",\"k_head_stride\":"<<f.k.strides(1)<<",\"k_token_stride\":"<<f.k.strides(2)
               <<",\"v_head_stride\":"<<f.v.strides(1)<<",\"v_token_stride\":"<<f.v.strides(2)
               <<",\"native_encodings\":1,\"output_elements_compared\":6144}\n"<<std::flush;
    }
    require(f.k.buffer().ptr()==k_identity && f.v.buffer().ptr()==v_identity,"Read changed input allocation identity");
    require(bits(f.k)==old_k && bits(f.v)==old_v,"Reader modified physical K/V or padding");
  }
}
void owner_case() {
  const int n=257;
  auto actual=[n]() {
    auto f=fixture(n,1); pg::PageTable local_table(f.ids,f.physical,n);
    auto q=head_major(24,1,1,false);
    auto mask=mask_array(masks(n).back(),n);
    // Every input and the page-table object leaves scope before eval. The
    // result graph, then MLX encoder, must retain the real array/buffer owners.
    return pg::read(q,f.k,f.v,local_table,f.kind,mask,gpu());
  }();
  auto q=head_major(24,1,1,false), k=head_major(2,n,n,false),v=head_major(2,n,n,true);
  auto expected=reference(q,k,v,mask_array(masks(n).back(),n));ready(actual,expected);
  equal_bits(actual,expected,"Input owners retained through lazy evaluation");
  std::cout<<"{\"event\":\"reader_owner_case\",\"passed\":true,\"tokens\":257}\n"<<std::flush;
}
void benchmark(int n,int iterations,const pg::IdentityPageTable& identity) {
  auto q=head_major(24,1,1,false);auto f=fixture(n,2);ready(f.k,f.v);
  auto mask=mask_array(masks(n).back(),n);
  auto warm_native=pg::read_identity(q,f.k,f.v,identity,mask,gpu());
  auto warm_stock=reference(q,f.k,f.v,mask);ready(warm_native,warm_stock);
  equal_bits(warm_native,warm_stock,"Benchmark numerical prerequisite");
  const char* order[] = {"stock_capacity_view","paged_identity_view","paged_identity_view","stock_capacity_view"};
  for (int leg=0;leg<4;++leg) {
    const bool native=leg==1||leg==2;mx::synchronize(gpu());const auto encoded=pg::encoded_reads();
    const auto started=std::chrono::steady_clock::now();
    for (int i=0;i<iterations;++i) {
      auto output=native ? pg::read_identity(q,f.k,f.v,identity,mask,gpu()) : reference(q,f.k,f.v,mask);
      if (asynchronous) {mx::async_eval(output);mx::synchronize(gpu());} else mx::eval(output);
      output.wait();
    }
    mx::synchronize(gpu());const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    const auto delta=pg::encoded_reads()-encoded;require(delta==uint64_t(native?iterations:0),"Benchmark encoding count mismatch");
    std::cout<<"{\"event\":\"reader_benchmark\",\"tokens\":"<<n<<",\"leg\":"<<leg
             <<",\"mode\":"<<std::quoted(order[leg])<<",\"iterations\":"<<iterations
             <<",\"wall_seconds\":"<<seconds<<",\"mean_call_seconds\":"<<seconds/iterations
             <<",\"native_encodings\":"<<delta
             <<",\"timed_readback\":false,\"timed_kv_gather_or_pack\":false,\"kernel_only_timing\":false}\n"<<std::flush;
  }
}
int main(int argc,char** argv) {
  std::cout<<std::boolalpha<<std::setprecision(17);
  try {
    bool run=false;int iterations=0;
    for (int i=1;i<argc;++i) {
      const std::string arg=argv[i];
      if (arg=="--run") run=true;
      else if (arg=="--async-eval") asynchronous=true;
      else if (arg=="--benchmark-iterations"&&i+1<argc) {
        const std::string value=argv[++i];size_t used=0;iterations=std::stoi(value,&used);
        require(used==value.size()&&iterations>=1&&iterations<=200,"Benchmark iterations outside 1...200");
      } else throw std::invalid_argument("Usage: probe --run [--async-eval] [--benchmark-iterations 1...200]");
    }
    if (!run) {std::cerr<<"Explicit --run is required; no GPU work was performed.\n";return 2;}
    pg::IdentityPageTable identity(16384);
    std::cout<<"{\"event\":\"start\",\"mlx_version\":"<<std::quoted(mx::version())
             <<",\"query_heads\":24,\"kv_heads\":2,\"head_dim\":256,\"query_tokens\":1,\"page_tokens\":32"
             <<",\"dtype\":\"bfloat16\",\"actual_qsa_indexer_executed\":false,\"model_loaded\":false"
             <<",\"identity_metadata_payload_bytes\":"<<identity.metadata_bytes()<<",\"async_eval\":"<<asynchronous<<"}\n"<<std::flush;
    for (int n:{31,32,33,255,256,257,1023,1024,1025,2051,2052,4095,4096,11232,11233})length_case(n,identity);
    owner_case();
    if (iterations) {benchmark(11232,iterations,identity);benchmark(11233,iterations,identity);}
    mx::synchronize(gpu());
    std::cout<<"{\"event\":\"summary\",\"passed\":true,\"lengths\":15,\"cases\":"<<cases
             <<",\"checks\":"<<checks<<",\"native_encodings\":"<<pg::encoded_reads()
             <<",\"model_correctness_verified\":false,\"actual_qsa_indexer_executed\":false}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr<<"paged reader probe failed: "<<error.what()<<"\n";
    std::cout<<"{\"event\":\"summary\",\"passed\":false,\"completed_cases\":"<<cases
             <<",\"error\":"<<std::quoted(error.what())<<"}\n";return 1;
  }
}
