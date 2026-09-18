"""Optional whole-layer HCWrite probe; unchanged S1 and learned-buffer names.

Standalone HCWrite parity does not imply whole-layer parity: an opaque call
materializes its FP16 injection/output inputs. This wrapper deliberately keeps
the original output contract so a device pair can detect that graph effect.
"""
import copy

from coreai_hc_write_probe import DirectHCWrite, get_kernel, POLICIES
from export_coreai_pd import DecoderLayer


class HCWriteLayer(DecoderLayer):
    def forward(self, stream, *values):
        use_attention = stream.shape[1] > 1 and self.hc_write_scope in ('attention', 'both')
        use_moe = stream.shape[1] > 1 and self.hc_write_scope in ('moe', 'both')
        if self.ple is not None:
            embedding, states = values[0], values[1:]
            stream, next_ple = self.ple(stream, embedding, states[-1])
            states = states[:-1]
        else:
            states = values
        mixed, injection = self.attention_read(stream)
        result = self.attention(mixed, *states)
        stream = (self.direct_hc_write if use_attention else self.write)(stream, result[0], injection)
        mixed, injection = self.moe_read(stream)
        output = self.moe(mixed)[0]
        stream = (self.direct_hc_write if use_moe else self.write)(stream, output, injection)
        updated = result[1:1 + self.state_count]
        if self.ple is not None:
            updated = (*updated, next_ple)
        return (stream, *updated)


def with_direct_hc_write(original, *, scope='attention', policy='unrounded'):
    """Return an aliasing layer wrapper plus its kernel; never mutate original.

    Scope may be attention, moe or both. Only token counts>1 use the candidate.
    Parameter/buffer names and bytes remain unchanged; the new child is weightless.
    """
    if type(original) is not DecoderLayer or scope not in ('attention', 'moe', 'both') or policy not in POLICIES:
        raise ValueError('Expected an unmodified DecoderLayer and explicit HCWrite scope/policy')
    result = copy.copy(original)
    result.__class__ = HCWriteLayer
    # add_module must not modify the original module registry through shallow copy.
    result._modules = original._modules.copy()
    result.hc_write_scope, result.hc_write_policy = scope, policy
    result.direct_hc_write = DirectHCWrite(policy)
    return result, [get_kernel(policy)]
