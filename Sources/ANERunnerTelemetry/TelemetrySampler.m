// Private IOReport ABI declarations checked against SiliconScope commit
// 3bfe69f524d33db8e2418536d0179fcbfc3dac00 (CIOReport/include/ktop_ioreport.h).
// This process only discovers, subscribes and reads. It never controls a model.
#import "TelemetrySampler.h"
#import <IOKit/IOKitLib.h>
#import <IOKit/storage/IOBlockStorageDriver.h>
#include <libproc.h>
#include <sys/resource.h>
#include <mach/mach_time.h>
#include <errno.h>
#include <limits.h>

typedef struct IOReportSubscriptionRef *IOReportSubscriptionRef;
extern CFMutableDictionaryRef IOReportCopyChannelsInGroup(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
extern IOReportSubscriptionRef IOReportCreateSubscription(void *, CFMutableDictionaryRef, CFMutableDictionaryRef *, uint64_t, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamples(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamplesDelta(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
extern void IOReportIterate(CFDictionaryRef, int (^)(CFDictionaryRef));
extern CFStringRef IOReportChannelGetGroup(CFDictionaryRef);
extern CFStringRef IOReportChannelGetSubGroup(CFDictionaryRef);
extern CFStringRef IOReportChannelGetChannelName(CFDictionaryRef);
extern CFStringRef IOReportChannelGetUnitLabel(CFDictionaryRef);
extern int IOReportChannelGetFormat(CFDictionaryRef);
extern long IOReportSimpleGetIntegerValue(CFDictionaryRef, int);
extern int IOReportStateGetCount(CFDictionaryRef);
extern uint64_t IOReportStateGetResidency(CFDictionaryRef, int);
extern CFStringRef IOReportStateGetNameForIndex(CFDictionaryRef, int);

static mach_timebase_info_data_t timebase;
static void initializeClock(void) { if (!timebase.denom) mach_timebase_info(&timebase); }
static uint64_t ticksToNS(uint64_t ticks) {
    initializeClock();
    return (uint64_t)(((__uint128_t)ticks * timebase.numer) / timebase.denom);
}
uint64_t ANETelemetryNowNS(void) { return ticksToNS(mach_absolute_time()); }
NSDictionary *ANETelemetryTimebase(void) {
    initializeClock();
    return @{@"clock": @"mach_absolute_time_nanoseconds", @"unit": @"ns",
             @"numer": @(timebase.numer), @"denom": @(timebase.denom)};
}
static id nullValue(void) { return [NSNull null]; }
static NSString *cfString(CFStringRef s) { return s ? (__bridge NSString *)s : @""; }
static NSDictionary *errorValue(NSString *operation, int code, NSString *message) {
    return @{@"operation": operation, @"code": @(code), @"message": message};
}
static id counterDelta(id before, id after) {
    if (![before isKindOfClass:NSNumber.class] || ![after isKindOfClass:NSNumber.class]) return nullValue();
    uint64_t a = [before unsignedLongLongValue], b = [after unsignedLongLongValue];
    return b >= a ? @(b - a) : nullValue();
}
static NSDictionary *channelIdentity(CFDictionaryRef channel) {
    NSString *group = cfString(IOReportChannelGetGroup(channel));
    NSString *subgroup = cfString(IOReportChannelGetSubGroup(channel));
    NSString *name = cfString(IOReportChannelGetChannelName(channel));
    return @{@"id": [@[group, subgroup, name] componentsJoinedByString:@"/"],
             @"group": group, @"subgroup": subgroup, @"name": name,
             @"format": @(IOReportChannelGetFormat(channel)),
             @"unit": cfString(IOReportChannelGetUnitLabel(channel))};
}

@implementation ANETelemetrySampler {
    pid_t _pid;
    uint64_t _originalStartTicks, _previousEndNS, _previousIOEndNS;
    BOOL _identityAvailable;
    NSString *_stopReason;
    NSDictionary *_metadata, *_previousProcess, *_previousDisks;
    IOReportSubscriptionRef _subscription;
    CFMutableDictionaryRef _subscribedChannels;
    CFDictionaryRef _previousIO;
    NSString *_ioSource, *_ioKind;
    NSMutableArray *_ioAttempts;
}
- (NSDictionary *)metadata { return _metadata; }
- (NSString *)stopReason { return _stopReason; }

- (instancetype)initWithPID:(pid_t)pid {
    if (!(self = [super init])) return nil;
    _pid = pid;
    struct rusage_info_v4 usage = {0};
    errno = 0;
    int rc = proc_pid_rusage(pid, RUSAGE_INFO_V4, (rusage_info_t *)&usage);
    int usageError = errno;
    _identityAvailable = rc == 0 && usage.ri_proc_start_abstime != 0;
    _originalStartTicks = usage.ri_proc_start_abstime;
    _ioAttempts = [NSMutableArray array];
    // One successful subscription for the sampler lifetime; no per-sample rediscovery.
    if (![self subscribeGroup:@"AMC Stats" subgroup:@"Perf Counters" kind:@"byte_counter_channels"]) {
        for (NSString *group in @[@"PMP0", @"PMP", @"PMP1"]) {
            if ([self subscribeGroup:group subgroup:@"DCS BW" kind:@"bandwidth_residency_histogram"]) break;
        }
    }
    _metadata = @{
        @"target_pid": @(pid), @"target_start_abstime": _identityAvailable ? @(_originalStartTicks) : nullValue(),
        @"target_start_ns": _identityAvailable ? @(ticksToNS(_originalStartTicks)) : nullValue(),
        @"target_identity_error": _identityAvailable ? nullValue() : errorValue(@"proc_pid_rusage", usageError, @"Cannot establish process identity; sampling will stop"),
        @"ioreport": @{@"source": _ioSource ?: @"unavailable", @"kind": _ioKind ?: @"unavailable",
                       @"estimated": @([_ioKind isEqualToString:@"bandwidth_residency_histogram"]),
                       @"subscription_attempts": _ioAttempts},
        @"process_source": @"libproc proc_pid_rusage(RUSAGE_INFO_V4) + proc_pidinfo(PROC_PIDTASKINFO)",
        @"process_scope": @"Target PID lifetime counters only; excludes children and unrelated processes",
        @"system_disk_source": @"IOKit IOBlockStorageDriver Statistics Bytes (Read)/Bytes (Write)",
        @"system_disk_scope": @"Device/global observed block-storage-driver counters, including other processes and this collector's log I/O; not target-process counters or confirmed physical NAND bytes",
        @"interval_semantics": @"start_ns=previous collection end; end_ns=current collection end. Each source also records its own read window. Crossing a model phase must not be apportioned by duration.",
        @"units": @{@"timestamps": @"ns", @"cpu_time_raw": @"mach_absolute_time ticks", @"cpu_time_converted": @"ns", @"disk_counters": @"bytes", @"memory": @"bytes", @"page_faults_and_pageins": @"count"},
        @"limitations": @[@"IOReport histograms contain bandwidth-labeled bins with raw events; no DRAM bytes or GB/s are inferred.",
                           @"Do not add AMCC aggregate to requestors, or RD+WR to RD and WR. Histogram floors/saturation remain uncalibrated.",
                           @"AMC byte channels are kept individually; requestor, cache and DRAM mapping is not inferred.",
                           @"Physical DRAM counters and GPU utilization are unavailable unless separately established.",
                           @"File-cache hits can make logical pread bytes differ from both process and device disk accounting.",
                           @"The system_disk aggregate is an observed-driver sum. Virtual/backing devices can overlap; per-device records are authoritative, not a physical-system total."]
    };
    return self;
}

- (BOOL)subscribeGroup:(NSString *)group subgroup:(NSString *)subgroup kind:(NSString *)kind {
    CFMutableDictionaryRef desired = IOReportCopyChannelsInGroup((__bridge CFStringRef)group, (__bridge CFStringRef)subgroup, 0, 0, 0);
    NSArray *channels = desired ? ((__bridge NSDictionary *)desired)[@"IOReportChannels"] : nil;
    NSMutableDictionary *attempt = [@{@"group": group, @"subgroup": subgroup,
        @"discovered_channels": @(channels.count), @"subscription_success": @NO} mutableCopy];
    [_ioAttempts addObject:attempt];
    if (!desired || !channels.count || channels.count > 4096) {
        attempt[@"error"] = errorValue(@"IOReportCopyChannelsInGroup", 0, @"No bounded channel set available");
        if (desired) CFRelease(desired);
        return NO;
    }
    CFMutableDictionaryRef subscribed = NULL;
    IOReportSubscriptionRef subscription = IOReportCreateSubscription(NULL, desired, &subscribed, 0, NULL);
    if (!subscription || !subscribed) {
        attempt[@"error"] = errorValue(@"IOReportCreateSubscription", 0, @"Subscription returned null at the current uid; privileged access was not attempted");
        // Private subscription ownership is not documented. Keep the bounded
        // attempted dictionaries until this isolated process exits; do not risk
        // releasing an input consumed by CreateSubscription.
        return NO;
    }
    _subscription = subscription;
    _subscribedChannels = subscribed;
    _ioSource = [@[group, subgroup] componentsJoinedByString:@"/"];
    _ioKind = kind;
    attempt[@"subscription_success"] = @YES;
    return YES;
}

- (NSDictionary *)readProcess {
    uint64_t start = ANETelemetryNowNS();
    struct rusage_info_v4 r = {0};
    errno = 0;
    int rc = proc_pid_rusage(_pid, RUSAGE_INFO_V4, (rusage_info_t *)&r), savedErrno = errno;
    NSMutableDictionary *result = [@{@"source": @"libproc", @"read_start_ns": @(start),
        @"rusage": nullValue(), @"taskinfo": nullValue(), @"error": nullValue()} mutableCopy];
    if (rc != 0) {
        result[@"error"] = errorValue(@"proc_pid_rusage", savedErrno, @(strerror(savedErrno)));
        _stopReason = savedErrno == ESRCH || savedErrno == ENOENT ? @"target_unavailable" : @"target_identity_unverifiable";
    } else if (!_identityAvailable || r.ri_proc_start_abstime != _originalStartTicks) {
        result[@"error"] = errorValue(@"process_identity", 0, @"Target PID identity unavailable or changed; refusing to sample a replacement process");
        _stopReason = @"target_identity_changed";
    } else {
        result[@"rusage"] = @{
            @"pid": @(_pid), @"start_abstime": @(r.ri_proc_start_abstime), @"exit_abstime": @(r.ri_proc_exit_abstime),
            @"disk_read_bytes_cumulative": @(r.ri_diskio_bytesread), @"disk_write_bytes_cumulative": @(r.ri_diskio_byteswritten),
            @"cpu_user_time_ticks_cumulative": @(r.ri_user_time), @"cpu_system_time_ticks_cumulative": @(r.ri_system_time),
            @"cpu_user_time_ns_cumulative": @(ticksToNS(r.ri_user_time)), @"cpu_system_time_ns_cumulative": @(ticksToNS(r.ri_system_time)),
            @"resident_bytes": @(r.ri_resident_size), @"physical_footprint_bytes": @(r.ri_phys_footprint),
            @"wired_bytes": @(r.ri_wired_size), @"pageins_cumulative": @(r.ri_pageins)
        };
        if (r.ri_proc_exit_abstime != 0) _stopReason = @"target_exited";
        struct proc_taskinfo task = {0};
        errno = 0;
        int size = proc_pidinfo(_pid, PROC_PIDTASKINFO, 0, &task, sizeof(task));
        if (size == sizeof(task)) {
            result[@"taskinfo"] = @{@"page_faults_cumulative": task.pti_faults >= 0 ? @(task.pti_faults) : nullValue(),
                @"pageins_cumulative": task.pti_pageins >= 0 ? @(task.pti_pageins) : nullValue(),
                @"cow_faults_cumulative": task.pti_cow_faults >= 0 ? @(task.pti_cow_faults) : nullValue(),
                @"virtual_bytes": @(task.pti_virtual_size), @"resident_bytes": @(task.pti_resident_size)};
        } else result[@"taskinfo_error"] = errorValue(@"proc_pidinfo", errno, @"Task info unavailable; rusage may remain available");
    }
    result[@"read_end_ns"] = @(ANETelemetryNowNS());
    return result;
}

- (NSDictionary *)readDisks {
    uint64_t start = ANETelemetryNowNS();
    NSMutableDictionary *devices = [NSMutableDictionary dictionary];
    NSMutableArray *errors = [NSMutableArray array];
    io_iterator_t iterator = IO_OBJECT_NULL;
    kern_return_t kr = IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOBlockStorageDriver"), &iterator);
    if (kr != KERN_SUCCESS) [errors addObject:errorValue(@"IOServiceGetMatchingServices", kr, @"Device inventory unavailable")];
    NSUInteger count = 0;
    io_object_t entry;
    while (kr == KERN_SUCCESS && (entry = IOIteratorNext(iterator))) {
        if (++count > 128) { IOObjectRelease(entry); [errors addObject:errorValue(@"disk_inventory", 0, @"Exceeded128-device bound")]; break; }
        uint64_t registryID = 0;
        kern_return_t idStatus = IORegistryEntryGetRegistryEntryID(entry, &registryID);
        io_name_t name = {0}; IORegistryEntryGetName(entry, name);
        io_name_t className = {0}; IOObjectGetClass(entry, className);
        io_string_t registryPath = {0};
        kern_return_t pathStatus = IORegistryEntryGetPath(entry, kIOServicePlane, registryPath);
        CFTypeRef property = IORegistryEntryCreateCFProperty(entry, CFSTR(kIOBlockStorageDriverStatisticsKey), kCFAllocatorDefault, 0);
        NSDictionary *stats = property && CFGetTypeID(property) == CFDictionaryGetTypeID() ? (__bridge NSDictionary *)property : nil;
        id read = stats[@kIOBlockStorageDriverStatisticsBytesReadKey];
        id write = stats[@kIOBlockStorageDriverStatisticsBytesWrittenKey];
        if (idStatus == KERN_SUCCESS && [read isKindOfClass:NSNumber.class] && [write isKindOfClass:NSNumber.class]) {
            devices[[NSString stringWithFormat:@"%llu", registryID]] = @{
                @"registry_id": @(registryID), @"name": @(name), @"class": @(className),
                @"registry_path": pathStatus == KERN_SUCCESS ? @(registryPath) : nullValue(),
                @"scope": @"device/global", @"unit": @"bytes",
                @"read_bytes_cumulative": read, @"write_bytes_cumulative": write};
        } else [errors addObject:errorValue(@"IOBlockStorageDriver Statistics", idStatus, [NSString stringWithFormat:@"Missing byte counters for %s", name])];
        if (property) CFRelease(property);
        IOObjectRelease(entry);
    }
    if (iterator) IOObjectRelease(iterator);
    if (!devices.count && !errors.count) [errors addObject:errorValue(@"disk_inventory", 0, @"No IOBlockStorageDriver byte counters found")];
    return @{@"source": @"IOKit/IOBlockStorageDriver/Statistics", @"scope": @"device/global; includes unrelated processes and collector; not target process",
             @"aggregate_kind": @"observed_driver_sum; virtual/backing devices may overlap",
             @"devices": devices, @"errors": errors, @"read_start_ns": @(start), @"read_end_ns": @(ANETelemetryNowNS())};
}

- (NSDictionary *)readIOReport {
    uint64_t start = ANETelemetryNowNS();
    BOOL histogram = [_ioKind isEqualToString:@"bandwidth_residency_histogram"];
    NSMutableDictionary *result = [@{@"source": _ioSource ?: @"unavailable", @"kind": _ioKind ?: @"unavailable",
        @"estimated": @(histogram), @"scope": @"system/requestor channels; not PID attributable",
        @"unit": histogram ? @"events" : @"bytes", @"channels": nullValue(), @"error": nullValue(),
        @"read_start_ns": @(start), @"delta_start_ns": _previousIO ? @(_previousIOEndNS) : nullValue()} mutableCopy];
    if (!_subscription) {
        result[@"unit"] = nullValue();
        result[@"error"] = errorValue(@"IOReport", 0, @"No subscription available");
    } else {
        CFDictionaryRef current = IOReportCreateSamples(_subscription, _subscribedChannels, NULL);
        uint64_t captureEnd = ANETelemetryNowNS();
        CFDictionaryRef delta = current && _previousIO ? IOReportCreateSamplesDelta(_previousIO, current, NULL) : NULL;
        NSMutableArray *channels = [NSMutableArray array];
        if (delta) {
            IOReportIterate(delta, ^int(CFDictionaryRef channel) {
                if (channels.count >= 4096) return 1;
                NSMutableDictionary *item = [channelIdentity(channel) mutableCopy];
                int format = [item[@"format"] intValue];
                if (histogram && format == 2) {
                    int count = IOReportStateGetCount(channel);
                    if (count < 0 || count > 256) {
                        item[@"error"] = errorValue(@"IOReportStateGetCount", count, @"State count outside bound");
                    } else {
                        NSMutableArray *names = [NSMutableArray array], *values = [NSMutableArray array];
                        for (int i = 0; i < count; i++) {
                            [names addObject:cfString(IOReportStateGetNameForIndex(channel, i))];
                            [values addObject:@(IOReportStateGetResidency(channel, i))];
                        }
                        item[@"state_names"] = names;
                        item[@"residency_delta_raw"] = values;
                        item[@"estimated"] = @YES;
                    }
                    [channels addObject:item];
                } else if (!histogram && format == 1 && [item[@"unit"] isEqual:@"B"]) {
                    long value = IOReportSimpleGetIntegerValue(channel, 0);
                    item[@"byte_delta"] = value >= 0 ? @(value) : nullValue();
                    if (value < 0) item[@"error"] = errorValue(@"IOReportSimpleGetIntegerValue", 0, @"Negative/reset/unpopulated byte counter");
                    [channels addObject:item];
                }
                return 0;
            });
            result[@"channels"] = channels;
            if (!channels.count) result[@"error"] = errorValue(@"IOReportIterate", 0, @"No supported channels in delta");
        } else result[@"error"] = errorValue(@"IOReportCreateSamplesDelta", 0, current && !_previousIO ? @"Baseline only; no preceding sample" : @"Sample or delta unavailable");
        if (delta) CFRelease(delta);
        if (_previousIO) CFRelease(_previousIO);
        _previousIO = current;
        _previousIOEndNS = captureEnd;
        result[@"delta_end_ns"] = @(captureEnd);
    }
    result[@"read_end_ns"] = @(ANETelemetryNowNS());
    return result;
}

- (NSDictionary *)sample:(NSUInteger)sampleID baseline:(BOOL)baseline finalPartial:(BOOL)partial {
    uint64_t start = ANETelemetryNowNS();
    // Revalidate process start time before each collection, before private API I/O.
    NSDictionary *process = [self readProcess];
    NSDictionary *io = [self readIOReport], *disks = [self readDisks];
    uint64_t end = ANETelemetryNowNS();
    NSMutableDictionary *result = [@{@"type": baseline ? @"baseline" : @"sample", @"schema_version": @1,
        @"clock": @"mach_absolute_time_nanoseconds", @"target_pid": @(_pid), @"sample_id": @(sampleID),
        @"start_ns": @(_previousEndNS ?: end), @"end_ns": @(end), @"collection_start_ns": @(start),
        @"is_final_partial": @(partial), @"process": process, @"system_disk": disks, @"ioreport": io,
        @"process_disk_read_bytes_delta": nullValue(), @"process_disk_write_bytes_delta": nullValue(),
        @"system_disk_read_bytes_delta": nullValue(), @"system_disk_write_bytes_delta": nullValue(),
        @"physical_dram_read_bytes_delta": nullValue(), @"physical_dram_write_bytes_delta": nullValue(),
        @"delta_errors": [NSMutableArray array], @"stop_reason": _stopReason ?: nullValue()} mutableCopy];
    NSMutableArray *errors = result[@"delta_errors"];
    id previousUsage = _previousProcess[@"rusage"], currentUsage = process[@"rusage"];
    if (!baseline && [previousUsage isKindOfClass:NSDictionary.class] && [currentUsage isKindOfClass:NSDictionary.class]) {
        for (NSString *direction in @[@"read", @"write"]) {
            NSString *key = [NSString stringWithFormat:@"disk_%@_bytes_cumulative", direction];
            id delta = counterDelta(previousUsage[key], currentUsage[key]);
            result[[NSString stringWithFormat:@"process_disk_%@_bytes_delta", direction]] = delta;
            if (delta == nullValue()) [errors addObject:errorValue(@"process_disk_delta", 0, @"Counter unavailable or decreased")];
        }
    } else [errors addObject:errorValue(@"process_disk_delta", 0, baseline ? @"Baseline only" : @"One process endpoint unavailable")];
    NSDictionary *previousDevices = _previousDisks[@"devices"], *currentDevices = disks[@"devices"];
    BOOL validDisks = !baseline && previousDevices.count && currentDevices.count &&
        [_previousDisks[@"errors"] count] == 0 && [disks[@"errors"] count] == 0 &&
        [[NSSet setWithArray:previousDevices.allKeys] isEqual:[NSSet setWithArray:currentDevices.allKeys]];
    for (NSString *direction in @[@"read", @"write"]) {
        uint64_t sum = 0; BOOL valid = validDisks;
        NSString *key = [NSString stringWithFormat:@"%@_bytes_cumulative", direction];
        for (NSString *deviceID in currentDevices) {
            id delta = counterDelta(previousDevices[deviceID][key], currentDevices[deviceID][key]);
            if (![delta isKindOfClass:NSNumber.class] || UINT64_MAX - sum < [delta unsignedLongLongValue]) { valid = NO; break; }
            sum += [delta unsignedLongLongValue];
        }
        if (valid) result[[NSString stringWithFormat:@"system_disk_%@_bytes_delta", direction]] = @(sum);
        else [errors addObject:errorValue(@"system_disk_delta", 0, baseline ? @"Baseline only" : @"Device set changed, read failed, or a counter decreased")];
    }
    _previousProcess = process; _previousDisks = disks; _previousEndNS = end;
    return result;
}

- (void)dealloc {
    if (_previousIO) CFRelease(_previousIO);
    // The one private subscription and bounded discovery dictionaries live
    // until process termination. No private teardown ownership is assumed.
}
@end
