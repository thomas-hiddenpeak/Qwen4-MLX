// Uses the same private IOReport ABI declarations as TelemetrySampler.m.
// State names and units are retained verbatim; P-state labels are not MHz.
#import "GPUStateSampler.h"
#import "TelemetrySampler.h"

typedef struct IOReportSubscriptionRef *IOReportSubscriptionRef;
extern CFMutableDictionaryRef IOReportCopyChannelsInGroup(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
extern IOReportSubscriptionRef IOReportCreateSubscription(void *, CFMutableDictionaryRef, CFMutableDictionaryRef *, uint64_t, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamples(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamplesDelta(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
extern void IOReportIterate(CFDictionaryRef, int (^)(CFDictionaryRef));
extern CFStringRef IOReportChannelGetChannelName(CFDictionaryRef);
extern CFStringRef IOReportChannelGetUnitLabel(CFDictionaryRef);
extern int IOReportChannelGetFormat(CFDictionaryRef);
extern int IOReportStateGetCount(CFDictionaryRef);
extern uint64_t IOReportStateGetResidency(CFDictionaryRef, int);
extern CFStringRef IOReportStateGetNameForIndex(CFDictionaryRef, int);

static id stateString(CFStringRef value) { return value ? (__bridge NSString *)value : (id)NSNull.null; }

@interface ANEGPUStateChannel : NSObject
@property(nonatomic, readonly) NSDictionary *metadata;
- (instancetype)initWithSubgroup:(NSString *)subgroup;
- (NSDictionary *)sample;
@end

@implementation ANEGPUStateChannel {
    NSString *_subgroup;
    NSDictionary *_metadata;
    IOReportSubscriptionRef _subscription;
    CFMutableDictionaryRef _channels;
    CFDictionaryRef _previous;
    uint64_t _previousStart, _previousEnd;
}
- (NSDictionary *)metadata { return _metadata; }
- (instancetype)initWithSubgroup:(NSString *)subgroup {
    if (!(self = [super init])) return nil;
    _subgroup = subgroup;
    CFMutableDictionaryRef desired = IOReportCopyChannelsInGroup(CFSTR("GPU Stats"), (__bridge CFStringRef)subgroup, 0, 0, 0);
    NSUInteger count = desired ? [((__bridge NSDictionary *)desired)[@"IOReportChannels"] count] : 0;
    // Bounds checked before subscribing; no privilege changes or rediscovery.
    if (count > 0 && count <= 32) {
        _subscription = IOReportCreateSubscription(NULL, desired, &_channels, 0, NULL);
    } else if (desired) {
        CFRelease(desired);
    }
    BOOL available = _subscription && _channels;
    _metadata = @{@"group": @"GPU Stats", @"subgroup": subgroup,
        @"discovered_channels": @(count), @"available": @(available),
        @"error": available ? (id)NSNull.null : @"Bounded subscription unavailable at current uid"};
    // Private subscription ownership is not a public ABI. Keep its bounded
    // dictionaries alive until this short-lived sidecar process exits.
    return self;
}
- (NSDictionary *)sample {
    uint64_t start = ANETelemetryNowNS();
    CFDictionaryRef current = (_subscription && _channels) ? IOReportCreateSamples(_subscription, _channels, NULL) : NULL;
    uint64_t end = ANETelemetryNowNS();
    CFDictionaryRef delta = current && _previous ? IOReportCreateSamplesDelta(_previous, current, NULL) : NULL;
    NSMutableArray *channels = [NSMutableArray array];
    __block BOOL valid = delta != NULL;
    if (delta) IOReportIterate(delta, ^int(CFDictionaryRef channel) {
        if (channels.count >= 32 || IOReportChannelGetFormat(channel) != 2) {
            valid = NO; return 0;
        }
        int count = IOReportStateGetCount(channel);
        if (count <= 0 || count > 256) {
            valid = NO; return 0;
        }
        NSMutableArray *states = [NSMutableArray array];
        for (int i = 0; i < count; i++) {
            uint64_t residency = IOReportStateGetResidency(channel, i);
            // Negative private-API sentinel represented as an unsigned value.
            if (residency > INT64_MAX) valid = NO;
            [states addObject:@{@"name": stateString(IOReportStateGetNameForIndex(channel, i)),
                               @"residency_delta_raw": @(residency)}];
        }
        [channels addObject:@{@"name": stateString(IOReportChannelGetChannelName(channel)),
            @"unit": stateString(IOReportChannelGetUnitLabel(channel)), @"states": states}];
        return 0;
    });
    valid = valid && channels.count > 0;
    NSDictionary *result = @{@"group": @"GPU Stats", @"subgroup": _subgroup,
        @"read_start_ns": @(start), @"read_end_ns": @(end),
        @"previous_read_start_ns": _previous ? @(_previousStart) : (id)NSNull.null,
        @"previous_read_end_ns": _previous ? @(_previousEnd) : (id)NSNull.null,
        @"complete_delta": @(valid), @"channels": delta ? channels : (id)NSNull.null,
        @"error": valid ? (id)NSNull.null : (!current ? @"Sample unavailable" : (!_previous ? @"Baseline only" : @"Invalid or missing state delta"))};
    if (delta) CFRelease(delta);
    if (_previous) CFRelease(_previous);
    // A missing endpoint breaks the chain; never span it with an implicit delta.
    _previous = current; _previousStart = start; _previousEnd = end;
    return result;
}
- (void)dealloc { if (_previous) CFRelease(_previous); }
@end

@implementation ANEGPUStateSampler {
    NSArray<ANEGPUStateChannel *> *_channels;
}
- (instancetype)init {
    if (!(self = [super init])) return nil;
    _channels = @[[[ANEGPUStateChannel alloc] initWithSubgroup:@"GPU Performance States"],
                 [[ANEGPUStateChannel alloc] initWithSubgroup:@"CLTM-induced GPU Performance States"]];
    return self;
}
- (NSDictionary *)metadata {
    NSMutableArray *sources = [NSMutableArray array];
    for (ANEGPUStateChannel *channel in _channels) [sources addObject:channel.metadata];
    return @{@"scope": @"System-wide; not attributable to target PID",
        @"sources": sources, @"thermal_source": @"NSProcessInfo.thermalState",
        @"limits": @"Raw named state residency only. No MHz, temperature, utilization or causal throttling inference. Exclude phase-crossing endpoint envelopes."};
}
- (NSDictionary *)sample {
    NSMutableArray *states = [NSMutableArray array];
    for (ANEGPUStateChannel *channel in _channels) [states addObject:[channel sample]];
    uint64_t start = ANETelemetryNowNS();
    NSProcessInfoThermalState state = NSProcessInfo.processInfo.thermalState;
    uint64_t end = ANETelemetryNowNS();
    NSString *name;
    switch (state) {
        case NSProcessInfoThermalStateNominal: name = @"nominal"; break;
        case NSProcessInfoThermalStateFair: name = @"fair"; break;
        case NSProcessInfoThermalStateSerious: name = @"serious"; break;
        case NSProcessInfoThermalStateCritical: name = @"critical"; break;
        default: name = @"unknown";
    }
    return @{@"scope": @"system-wide", @"state_channels": states,
        @"thermal": @{@"read_start_ns": @(start), @"read_end_ns": @(end),
                      @"state": name, @"raw_value": @(state)}};
}
@end
