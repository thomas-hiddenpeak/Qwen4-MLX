import ANERunnerCore
import ANERunnerGPU
import CMLX
import Darwin
import Foundation

extension RunnerCLI {
    /// Real existing-format1...2GiB checkpoint restoration. No streaming API,
    /// no paged backend, no old cache directory, no default limit change.
    static func probeGPULargeSSDImport(_ args: Arguments) throws {
        try args.validate(["--model-dir","--tokens-file","--output","--cache-dir"])
        let output = try args.require("--output")
        let directory = URL(fileURLWithPath:try args.require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
        let cache = URL(fileURLWithPath:try args.require("--cache-dir")).standardizedFileURL
        guard !FileManager.default.fileExists(atPath:output),
              !FileManager.default.fileExists(atPath:cache.path) else {
            throw CLIError.usage("Large SSD probe requires new output and a new dedicated cache directory")
        }
        let tokenURL = URL(fileURLWithPath:try args.require("--tokens-file")).standardizedFileURL
        let data = try Data(contentsOf:tokenURL), prompt: [Int32]
        if let plain = try? JSONDecoder().decode([Int32].self,from:data) { prompt=plain }
        else {
            struct TokenReport: Decodable { let tokens: [Int32] }
            prompt=try JSONDecoder().decode(TokenReport.self,from:data).tokens
        }
        guard prompt.count == 65534 else { throw CLIError.usage("Expected actual65534-token fixture") }
        let boundary=65312, gib=1024*1024*1024
        let cancellation = QwenCancellation()
        let oldTerm=Darwin.signal(SIGTERM,SIG_IGN), oldInt=Darwin.signal(SIGINT,SIG_IGN)
        let term=DispatchSource.makeSignalSource(signal:SIGTERM,queue:.global())
        let interrupt=DispatchSource.makeSignalSource(signal:SIGINT,queue:.global())
        term.setEventHandler { cancellation.cancel() }; interrupt.setEventHandler { cancellation.cancel() }
        term.resume(); interrupt.resume()
        defer { term.cancel(); interrupt.cancel(); Darwin.signal(SIGTERM,oldTerm); Darwin.signal(SIGINT,oldInt) }
        func object<T:Encodable>(_ value:T) throws -> Any {
            try JSONSerialization.jsonObject(with:JSONEncoder().encode(value))
        }
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        var report:[String:Any] = ["schema":"qwen-large-ssd-import-probe-v1","complete":false,"passed":false,
            "command":CommandLine.arguments,"cache_directory":cache.path,
            "binary_sha256":try MoETilingBytes.hash(URL(fileURLWithPath:CommandLine.arguments[0])),
            "tokens_file_sha256":MoETilingBytes.digest(data),"prompt_tokens":65534,"output_budget":2,
            "checkpoint_tokens":boundary,"state_budget_bytes":12*gib,"ram_cache_bytes":2*gib,
            "disk_pending_bytes":2*gib,"disk_capacity_bytes":4*gib,
            "notes":["Same48-layer model, two sequential generators with separate RAM caches and one retained SSD store.",
                "First cold generation is the full state/output oracle. Second request must import from disk, not cold fallback.",
                "Observer stores only host/hash anchors; no persistent Tensor aliases or complete host-state payload.",
                "Existing whole-Data archive path, bounded to a legal1...2GiB payload. Not262K SSD or streaming support.",
                "Diagnostic runtime/allocator observations are not uninstrumented throughput or RSS.",
                "SIGINT/SIGTERM request cooperative cancellation. GPU work joins before cleanup; SSD close has one30s bound."]]
        var rows=[[String:Any]](), states=[[String:Any]]()
        func write() throws { report["trials"]=rows;report["states"]=states;try emit(report,to:output) }
        try write()
        let disk = try QwenPrefixDiskStore(directory:cache,limits:.init(maxEntries:2,maxBytes:4*gib,
            maxPendingJobs:1,maxPendingBytes:2*gib))
        var closed=false
        var closeOutcome:QwenPrefixDiskCloseResult?
        func close() -> QwenPrefixDiskCloseResult {
            if let closeOutcome { return closeOutcome }
            let result=disk.close(drain:true,timeout:30);closed=true;closeOutcome=result
            report["close"]=["io_completed":result.ioCompleted,"callbacks_completed":result.callbacksCompleted,
                "completed":result.completed,"timeout_seconds":30]
            return result
        }
        defer { if !closed { _=close() } }
        func waitForIO(_ stage:String,budget:QwenStateBudget?=nil,requireReleased:Bool=false) throws {
            let deadline=now()+60_000_000_000
            while true {
                try cancellation.check()
                let s=disk.statistics
                if s.pendingJobs == 0 && s.pendingBytes == 0 && (s.foregroundReadIntents ?? 0) == 0 &&
                    (budget?.statistics.workspaceBytes ?? 0) == 0 &&
                    (!requireReleased || (budget?.statistics.totalBytes == 0 && budget?.statistics.currentLeases == 0)) { return }
                guard now()<deadline else { throw CLIError.usage("Timed out waiting for SSD "+stage) }
                Thread.sleep(forTimeInterval:0.02)
            }
        }
        do {
            let model=try QwenModel(modelDirectory:directory,stateBudgetBytes:12*gib)
            let checkpointBytes=try model.estimatedPrefixStateBytes(at:boundary)
            guard checkpointBytes>gib,checkpointBytes<2*gib else {
                throw CLIError.usage("Checkpoint must exercise the legal above1GiB import path")
            }
            report["checkpoint_logical_bytes"]=checkpointBytes
            report["memory_after_load"]=try MX.memory()
            let request=QwenGenerationRequest(tokens:prompt,maxTokens:2,contextLimit:65536,
                prefillChunk:416,mtpDepth:0,prefixCacheMaxTokens:boundary)
            try request.validate(configuration:model.configuration)
            var anchors=[Int:CacheReliabilityAnchor](), oracle:QwenGenerationResult?
            func anchor(_ state:QwenModel.State) throws -> CacheReliabilityAnchor {
                guard !state.hasPagedKV,let lease=model.stateBudget.reserve(bytes:256*1024*1024,kind:.workspace) else {
                    throw CLIError.usage("Cannot admit dense one-tensor diagnostic workspace")
                }
                defer { lease.release() }
                do {
                    let value=try CacheReliabilityAnchor(state)
                    try MX.synchronize()
                    guard value.valid,value.tensors.count==121 else { throw CLIError.usage("Invalid121-state anchor") }
                    return value
                } catch { let original=error;try MX.synchronize();throw original }
            }
            do {
                for round in 0..<2 {
                    try cancellation.check()
                    let label=round==0 ? "cold_populate" : "fresh_generator_disk_restore"
                    let before=disk.statistics
                    do {
                        let generator=try QwenGenerator(model:model,
                            prefixCacheLimits:.init(maxEntries:1,maxBytes:2*gib,diskRestoreTimeoutSeconds:60),
                            prefixDiskStore:disk)
                        guard generator.prefixCacheStatistics?.entries==0 else { throw CLIError.usage("New generator RAM is not empty") }
                        var events=[String](), callbackIDs=[Int32]()
                        func observe(_ event:String,_ state:QwenModel.State) throws {
                            try cancellation.check()
                            guard [boundary,65534,65535].contains(state.offset) else {
                                throw CLIError.usage("Unexpected observer offset")
                            }
                            let actual=try anchor(state)
                            if round==0 && anchors[state.offset]==nil { anchors[state.offset]=actual }
                            guard let expected=anchors[state.offset] else { throw CLIError.usage("Missing cold state oracle") }
                            let exact=actual.matches(expected)
                            states.append(["trial":label,"event":event,"offset":state.offset,
                                "oracle":"cold_populate:\(state.offset)","actual":try object(actual),
                                "expected":try object(expected),"passed":exact])
                            events.append(event);try write()
                            guard exact else { throw CLIError.usage("Complete mixed state differs after SSD restore") }
                        }
                        generator.prefixStateObserver=observe
                        generator.decodeStateObserver=observe
                        defer { generator.prefixStateObserver=nil;generator.decodeStateObserver=nil }
                        FileHandle.standardError.write(Data("Large SSD import: \(label)\n".utf8))
                        let started=now()
                        let result=try generator.generate(request,cancellation:cancellation,onToken:{ callbackIDs.append($0) })
                        if round==0 { oracle=result }
                        guard let oracle,let phase=result.phases?.prefill,let cacheStats=generator.prefixCacheStatistics else {
                            throw CLIError.usage("Missing result/cache statistics")
                        }
                        let cached=round==0 ? 0 : boundary
                        let correct=result.tokens==oracle.tokens && result.finishReason==oracle.finishReason &&
                            result.tokens.count==2 && callbackIDs==result.tokens && result.finishReason.rawValue=="length" &&
                            result.statistics.decodeRounds==1 && result.statistics.decodedTokenCount==1 &&
                            result.statistics.finalStateOffset==65535 && phase.cachedTokenCount==cached &&
                            phase.computedTokenCount==65534-cached && phase.cacheSource==(round==0 ? "cold" : "disk") &&
                            cacheStats.restoreFailures==0 && cacheStats.diskFallbacks==0 &&
                            (round==0 ? events.contains("coldBoundary") : events.contains("restore") && cacheStats.diskHits==1) &&
                            events.contains("firstToken") && events.contains("decode")
                        rows.append(["label":label,"result":try object(result),"cache":try object(cacheStats),
                            "events":events,"callback_ids":callbackIDs,"wall_seconds":Double(now()-started)*1e-9,
                            "memory":try MX.memory(),"passed":correct])
                        try write()
                        guard correct else { throw CLIError.usage("Large SSD import result/phase/cache contract failed") }
                        try waitForIO(label,budget:model.stateBudget)
                        try generator.clearPrefixCache()
                        // clear only RAM; a valid persisted archive must survive.
                        guard generator.prefixCacheStatistics?.entries==0 else { throw CLIError.usage("RAM clear failed") }
                    }
                    try MX.synchronize();try waitForIO(label+" cleanup",budget:model.stateBudget,requireReleased:true)
                    let after=disk.statistics, budget=model.stateBudget.statistics
                    let retained=after.entries==1 && after.corruptions==0 && after.writeFailures==0 &&
                        after.evictions==0 && !after.storageUnavailable &&
                        (round==0 ? after.published==1 && after.bytesWritten>gib : after.hits>before.hits && after.bytesRead-before.bytesRead>gib)
                    report[label+"_disk"]=try object(after)
                    report[label+"_budget"]=try object(budget)
                    try write()
                    guard retained,budget.totalBytes==0,budget.currentLeases==0 else {
                        throw CLIError.usage("Archive retention or resource cleanup failed")
                    }
                }
                guard Set(anchors.keys)==Set([boundary,65534,65535]) else { throw CLIError.usage("Incomplete cold state oracle") }
                try MX.synchronize();try waitForIO("final",budget:model.stateBudget,requireReleased:true)
                let closeResult=close()
                report["final_disk"]=try object(disk.statistics)
                report["final_budget"]=try object(model.stateBudget.statistics)
                guard closeResult.completed,model.stateBudget.statistics.totalBytes==0,
                      model.stateBudget.statistics.currentLeases==0 else { throw CLIError.usage("Bounded close did not drain resources") }
                report["complete"]=true;report["passed"]=true;try write()
            } catch {
                let original=error
                try MX.synchronize()
                _=close()
                report["error_budget"]=try object(model.stateBudget.statistics)
                throw original
            }
        } catch {
            if !closed { _=close() }
            report["error"]=String(describing:error);report["cancelled"]=cancellation.isCancelled
            report["passed"]=false;try? write();throw error
        }
    }
}
