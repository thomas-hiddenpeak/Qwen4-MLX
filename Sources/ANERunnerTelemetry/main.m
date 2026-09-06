#import "TelemetrySampler.h"
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>
#include <time.h>

static volatile sig_atomic_t stoppingSignal = 0;
static void onSignal(int value) { stoppingSignal = value; }
static BOOL parsePositive(const char *text, uint64_t maximum, uint64_t *out) {
    if (!text || !*text) return NO;
    for (const char *p = text; *p; p++) if (*p < '0' || *p > '9') return NO;
    errno = 0; char *end = NULL; unsigned long long value = strtoull(text, &end, 10);
    if (errno || !end || *end || value == 0 || value > maximum) return NO;
    *out = value; return YES;
}
static BOOL writeJSON(FILE *output, NSDictionary *object) {
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:object options:NSJSONWritingSortedKeys error:&error];
    if (!data || data.length > 4 * 1024 * 1024) {
        fprintf(stderr, "Telemetry JSON record unavailable or exceeds4MiB: %s\n", error.description.UTF8String ?: "record bound");
        return NO;
    }
    if (fwrite(data.bytes, 1, data.length, output) != data.length || fputc('\n', output) == EOF || fflush(output) != 0) {
        fprintf(stderr, "Telemetry output write failed: %s\n", strerror(errno)); return NO;
    }
    return YES;
}
static void usage(void) {
    fprintf(stderr, "ane-telemetry --output NEW_PATH --pid PID [--interval-ms N] [--max-samples N]\n"
        "Read-only JSONL sidecar. Default200ms,300 interval samples (plus metadata/baseline/summary).\n"
        "Interval1..60000ms; samples1..100000. SIGINT/SIGTERM writes a final partial sample.\n"
        "Target identity is checked on every sample; target disappearance/reuse ends sampling.\n");
}
int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc == 2 && strcmp(argv[1], "--help") == 0) { usage(); return 0; }
        NSString *path = nil; uint64_t pid = 0, interval = 200, maximum = 300;
        NSMutableSet *seen = [NSMutableSet set];
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) { usage(); return 2; }
            NSString *key = @(argv[i]);
            if ([seen containsObject:key]) { fprintf(stderr, "Duplicate option\n"); return 2; }
            [seen addObject:key];
            BOOL valid = YES;
            if ([key isEqualToString:@"--output"]) path = @(argv[i + 1]);
            else if ([key isEqualToString:@"--pid"]) valid = parsePositive(argv[i + 1], INT_MAX, &pid);
            else if ([key isEqualToString:@"--interval-ms"]) valid = parsePositive(argv[i + 1], 60000, &interval);
            else if ([key isEqualToString:@"--max-samples"]) valid = parsePositive(argv[i + 1], 100000, &maximum);
            else valid = NO;
            if (!valid) { fprintf(stderr, "Invalid option/value: %s\n", argv[i]); return 2; }
        }
        if (!pid || !path.length) { usage(); return 2; }
        int fd = open(path.fileSystemRepresentation, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd < 0) { fprintf(stderr, "Cannot create new telemetry output: %s\n", strerror(errno)); return 2; }
        FILE *output = fdopen(fd, "w");
        if (!output) { close(fd); fprintf(stderr, "Cannot open telemetry stream\n"); return 2; }
        struct sigaction action = {0}; action.sa_handler = onSignal; sigemptyset(&action.sa_mask);
        sigaction(SIGINT, &action, NULL); sigaction(SIGTERM, &action, NULL); signal(SIGPIPE, SIG_IGN);
        ANETelemetrySampler *sampler = [[ANETelemetrySampler alloc] initWithPID:(pid_t)pid];
        NSMutableDictionary *metadata = [sampler.metadata mutableCopy];
        [metadata addEntriesFromDictionary:@{@"type": @"metadata", @"schema_version": @1,
            @"clock": @"mach_absolute_time_nanoseconds", @"timebase": ANETelemetryTimebase(),
            @"sampler_pid": @(getpid()), @"uid": @(getuid()), @"created_ns": @(ANETelemetryNowNS()),
            @"created_utc": [[NSISO8601DateFormatter new] stringFromDate:[NSDate date]],
            @"interval_ms": @(interval), @"max_samples": @(maximum),
            @"bounded_record_bytes": @(4 * 1024 * 1024), @"operating_system": NSProcessInfo.processInfo.operatingSystemVersionString}];
        BOOL ok = writeJSON(output, metadata);
        if (ok) ok = writeJSON(output, [sampler sample:0 baseline:YES finalPartial:NO]);
        uint64_t count = 0;
        while (ok && !sampler.stopReason && count < maximum) {
            uint64_t wake = ANETelemetryNowNS() + interval * 1000000;
            while (!stoppingSignal && ANETelemetryNowNS() < wake) {
                uint64_t left = wake - ANETelemetryNowNS();
                // A second clock read could cross the deadline. Avoid unsigned
                // underflow becoming a many-year sleep.
                if (left > interval * 1000000) break;
                struct timespec delay = {.tv_sec = left / 1000000000, .tv_nsec = left % 1000000000};
                if (nanosleep(&delay, NULL) != 0 && errno != EINTR) break;
            }
            @autoreleasepool {
                count++;
                ok = writeJSON(output, [sampler sample:(NSUInteger)count baseline:NO finalPartial:stoppingSignal != 0]);
            }
            if (stoppingSignal) break;
        }
        NSString *reason = sampler.stopReason ?: (stoppingSignal ? @"signal" : @"max_samples");
        if (ok) ok = writeJSON(output, @{@"type": @"summary", @"schema_version": @1,
            @"clock": @"mach_absolute_time_nanoseconds", @"end_ns": @(ANETelemetryNowNS()),
            @"target_pid": @(pid), @"samples_written": @(count), @"stop_reason": reason,
            @"signal": @(stoppingSignal), @"target_signalled_by_sampler": @NO});
        if (fclose(output) != 0) ok = NO;
        return ok ? 0 : 1;
    }
}
