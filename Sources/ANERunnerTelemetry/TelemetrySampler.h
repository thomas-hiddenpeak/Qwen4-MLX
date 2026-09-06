#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// All timestamps use the same boot-relative clock as the runner phase markers.
uint64_t ANETelemetryNowNS(void);
NSDictionary *ANETelemetryTimebase(void);

@interface ANETelemetrySampler : NSObject
- (instancetype)initWithPID:(pid_t)pid;
@property(nonatomic, readonly) NSDictionary *metadata;
@property(nonatomic, readonly, nullable) NSString *stopReason;
- (NSDictionary *)sample:(NSUInteger)sampleID baseline:(BOOL)baseline finalPartial:(BOOL)partial;
@end

NS_ASSUME_NONNULL_END
