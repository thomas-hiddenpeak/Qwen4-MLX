#import <Foundation/Foundation.h>

/// Optional system-wide raw GPU state observations. No frequency mapping.
@interface ANEGPUStateSampler : NSObject
@property(nonatomic, readonly) NSDictionary *metadata;
- (NSDictionary *)sample;
@end
