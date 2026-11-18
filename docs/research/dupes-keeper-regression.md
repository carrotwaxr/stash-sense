# Duplicate keeper regression (#130)

This is the keeper suggestion (`api/duplicates/keeper.py`) scored against the owner's recorded picks. The set is `api/tests/fixtures/dupes_keeper_regression.jsonl`: one record per decided pair, anonymized to codec, resolution, bit rate, frame rate, duration, size, container and the two upscale flags (no names, paths or hashes). The two files of each record are keyed `a` and `b`.

Outcomes:

- **agree**: duplicate pairs: the suggestion keeps exactly the file the owner kept. Contained pairs: the suggestion keeps the short file only when the owner did (`keep_short`), and drops it when the owner chose `long_better`.
- **superset**: duplicate pairs only: the suggestion keeps the owner's file plus the other one (an upscale pair or a longer cut). Nothing the owner wanted is deleted, so this counts as agreement.
- **tie**: the suggestion flagged the pair as practically equal and kept the other file. Either is right, so ties are left out of the agreement rate.
- **deferred**: contained pairs only: the suggestion says to split the longer file first (`split_first`: the long file is at least 1.5x the short one, so it probably holds several scenes) and keeps both files, while the owner deleted the short one. The owner's picks predate splitting (#131), so the suggestion defers instead of disagreeing. Like ties, deferred records are left out of the agreement rate. When the owner kept the short file, a `split_first` suggestion is an **agree**.
- **disagree**: everything else. Every one is listed below with the suggestion's reason.

For contained pairs the short side is the one with the larger cover. The keeper decides by duration; where the two differ the reason is marked "short side by duration differs from cover".

Agreement is (agree + superset) / (records - tie - deferred).

## Totals

| Kind | Records | agree | superset | tie | deferred | disagree | Agreement |
|---|---|---|---|---|---|---|---|
| duplicate | 385 | 278 | 55 | 16 | 0 | 36 | 333/369 (90.2%) |
| contained | 311 | 193 | 0 | 0 | 113 | 5 | 193/198 (97.5%) |
| all | 696 | 471 | 55 | 16 | 113 | 41 | 526/567 (92.8%) |

## Disagreements (41)

| Record | Kind | Side A | Side B | Human | Suggestion | Reason |
|---|---|---|---|---|---|---|
| 6dbf3af76a1d | contained | h264 640x480 1.5 Mbps 30 fps 12.9 min mp4 | h264 640x480 1.5 Mbps 30 fps 16.9 min mkv | keep short (A) too | delete short (A) | Its content is inside the longer file at equal or better quality. |
| 8a299c91ad06 | contained | av1 1312x960 2.0 Mbps 30 fps 15.6 min mp4 | h264 540x360 0.7 Mbps 30 fps 22.3 min mp4 | keep short (A) too | delete short (A) | Its content is inside the longer file at equal or better quality. |
| 8ec03211fdf0 | contained | mpeg4 608x352 1.5 Mbps 29.97 fps 28.0 min avi | av1 1920x1080 11.7 Mbps 29.97 fps 37.5 min mp4 | delete short (A) | keep short (A) too | The shorter file is better quality and the longer one holds extra content, so both are kept. |
| 96e52fdef3c7 | contained | mpeg4 480x360 1.6 Mbps 29.97 fps 12.8 min mkv | h264 480x360 1.0 Mbps 29.97 fps 16.4 min mp4 | keep short (A) too | delete short (A) | Its content is inside the longer file at equal or better quality. |
| be37fb664ea0 | contained | mpeg4 576x320 1.0 Mbps 29.97 fps 48.7 min avi | mpeg4 512x384 1.0 Mbps 29.97 fps 67.0 min avi | keep short (A) too | delete short (A) | Its content is inside the longer file at equal or better quality. |
| 068f08177620 | duplicate | h264 640x368 1.2 Mbps 29.97 fps 49.5 min mp4 | h264 640x480 0.7 Mbps 25 fps 49.5 min mp4 | keep A | keep B | Keeps the best copy: more effective detail (640x480 at 0.7 Mbps beats 640x368 at 1.2 Mbps) |
| 0eecb3c853bc | duplicate | h264 1920x1080 12.3 Mbps 29.97 fps 30.6 min mp4 | hevc 1920x1080 5.0 Mbps 29.97 fps 30.8 min mp4 | keep B | keep A | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) |
| 16511487bec1 | duplicate | h264 720x540 1.8 Mbps 25 fps 9.0 min mp4 | h264 720x576 1.4 Mbps 25 fps 8.7 min mp4 | keep B | keep A | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) |
| 177e1d6e9d87 | duplicate | h264 852x480 1.1 Mbps 30 fps 63.1 min mp4 | wmv3 320x240 0.9 Mbps 29.97 fps 64.8 min wmv | keep B | keep A | Keeps the best copy: more effective detail (852x480 at 1.1 Mbps beats 320x240 at 0.9 Mbps) |
| 18b3616f6e8e | duplicate | h264 720x540 1.9 Mbps 25 fps 21.8 min mp4 | h264 1828x1080 5.1 Mbps 30 fps 23.8 min mp4 | keep A | keep B | Keeps the best copy: more effective detail (1828x1080 at 5.1 Mbps beats 720x540 at 1.9 Mbps) |
| 18f9cce95386 | duplicate | h264 768x432 1.6 Mbps 23.98 fps 40.3 min mp4 | h264 720x576 2.1 Mbps 30 fps 39.4 min mp4 | keep A | keep B | Keeps the best copy: more effective detail (720x576 at 2.1 Mbps beats 768x432 at 1.6 Mbps) |
| 1e0eac060e69 | duplicate | h264 720x540 1.9 Mbps 25 fps 23.3 min mp4 | h264 1440x1080 8.1 Mbps 59.94 fps 25.8 min mp4 | keep A | keep B | Keeps the best copy: more effective detail (1440x1080 at 8.1 Mbps beats 720x540 at 1.9 Mbps) |
| 211b05c5ae96 | duplicate | h264 640x480 1.2 Mbps 29.96 fps 21.2 min mp4 | wmv2 720x480 2.5 Mbps 29.97 fps 21.1 min wmv | keep B | keep A | Keeps the best copy: same picture; the better container |
| 2e2e9fe4590f | duplicate | h264 852x480 1.5 Mbps 30 fps 27.7 min mp4 | mpeg4 544x304 1.8 Mbps 29.97 fps 27.8 min avi | keep B | keep A | Keeps the best copy: more effective detail (852x480 at 1.5 Mbps beats 544x304 at 1.8 Mbps) |
| 339ac1d59795 | duplicate | h264 896x672 6.2 Mbps 29.97 fps 19.9 min mp4 | h264 1280x720 6.3 Mbps 29.97 fps 19.7 min mp4 | keep A | keep B | Keeps the best copy: more effective detail (1280x720 at 6.3 Mbps beats 896x672 at 6.2 Mbps) |
| 36e8583a0c5a | duplicate | h264 640x368 1.2 Mbps 29.97 fps 33.5 min mp4 | h264 1920x1080 3.0 Mbps 29.97 fps 33.6 min mp4 | keep A | keep B | Keeps the best copy: more effective detail (1920x1080 at 3.0 Mbps beats 640x368 at 1.2 Mbps) |
| 3f3518cc0fd2 | duplicate | h264 720x540 1.8 Mbps 25 fps 18.3 min mp4 | h264 720x576 1.4 Mbps 25 fps 18.0 min mp4 | keep B | keep A | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) |
| 42bd7131a5fe | duplicate | h264 640x480 2.1 Mbps 29.97 fps 39.7 min mp4 | mpeg4 512x384 1.4 Mbps 29.97 fps 39.7 min avi | keep B | keep A | Keeps the best copy: more effective detail (640x480 at 2.1 Mbps beats 512x384 at 1.4 Mbps) |
| 46d5cbbeaba0 | duplicate | h264 640x480 2.1 Mbps 29.97 fps 19.0 min mp4 | wmv2 720x480 2.5 Mbps 29.97 fps 18.8 min wmv | keep B | keep A | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) |
| 478f715d7a12 | duplicate | h264 656x480 1.2 Mbps 29.97 fps 20.9 min mp4 | wmv2 640x480 2.1 Mbps 23.98 fps 21.6 min wmv | keep B | keep A | Keeps the best copy: same picture at a higher frame rate (29.97 fps) |
| 4af72f7431fb | duplicate | h264 640x480 1.3 Mbps 30 fps 27.4 min mp4 | h264 640x480 0.8 Mbps 29.97 fps 27.6 min mp4 | keep B | keep A | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) |
| 4dc1f94bf193 | duplicate | h264 1920x1080 12.2 Mbps 29.97 fps 41.0 min mp4 | h264 1920x1080 4.4 Mbps 29.97 fps 41.1 min mp4 | keep B | keep A | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) |
| 6dc04b3b9a2c | duplicate | h264 1280x720 4.1 Mbps 29.97 fps 39.6 min mp4 | h264 1120x832 10.3 Mbps 29.97 fps 39.4 min mp4 | keep A | keep B | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) |
| 6e0587ac3351 | duplicate | mpeg4 512x384 1.3 Mbps 25 fps 19.2 min avi | h264 640x480 0.9 Mbps 24 fps 19.2 min mkv | keep A | keep B | Keeps the best copy: more effective detail (640x480 at 0.9 Mbps beats 512x384 at 1.3 Mbps) |
| 778d66b519a4 | duplicate | h264 720x540 1.9 Mbps 25 fps 25.4 min mp4 | h264 720x404 1.9 Mbps 25 fps 26.1 min mp4 | keep B | keep A | Keeps the best copy: more effective detail (720x540 at 1.9 Mbps beats 720x404 at 1.9 Mbps) |
| 881ff6292c03 | duplicate | h264 720x480 1.3 Mbps 29.97 fps 12.6 min mp4 | h264 656x480 1.7 Mbps 30 fps 12.0 min mp4 | keep A | keep B | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) |
| 941530bf6f82 | duplicate | wmv3 640x480 1.6 Mbps 0 fps 19.9 min wmv | mpeg4 640x480 2.6 Mbps 29.97 fps 19.1 min avi | keep A | keep B | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) |
| 98f9a13e664b | duplicate | h264 720x540 2.1 Mbps 29.97 fps 28.1 min mp4 | mpeg2video 720x480 5.5 Mbps 29.97 fps 28.1 min mpg | keep A | keep B | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) |
| a46a478c7fb0 | duplicate | h264 640x480 1.2 Mbps 29.97 fps 40.1 min mp4 | mpeg4 640x480 2.3 Mbps 29.97 fps 40.2 min avi | keep B | keep A | Keeps the best copy: same picture; the better container |
| a5ba237a548a | duplicate | h264 720x540 1.9 Mbps 25 fps 25.8 min mp4 | h264 720x576 1.4 Mbps 25 fps 24.4 min mp4 | keep B | keep A | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) |
| aa5cf486bb40 | duplicate | mpeg4 512x288 0.9 Mbps 29.97 fps 19.3 min avi | wmv2 720x480 2.6 Mbps 0 fps 19.3 min wmv | keep A | keep B | Keeps the best copy: more effective detail (720x480 at 2.6 Mbps beats 512x288 at 0.9 Mbps) |
| be2788993325 | duplicate | h264 640x480 0.8 Mbps 29.97 fps 35.5 min mp4 | h264 640x480 1.2 Mbps 29.97 fps 35.2 min mp4 | keep A | keep B | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) |
| bf521fc12287 | duplicate | h264 640x480 1.2 Mbps 29.96 fps 27.6 min mp4 | h264 720x480 0.8 Mbps 29.97 fps 27.7 min mp4 | keep B | keep A | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) |
| c0a150324e69 | duplicate | h264 720x576 1.4 Mbps 25 fps 10.3 min mp4 | h264 720x540 1.9 Mbps 25 fps 10.1 min mp4 | keep A | keep B | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) |
| c8f7048f6d07 | duplicate | h264 720x528 1.3 Mbps 29.97 fps 20.3 min mp4 | h264 720x480 1.5 Mbps 29.97 fps 19.9 min mp4 | keep A | keep B | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) |
| cfae2a38816e | duplicate | h264 720x480 1.3 Mbps 29.97 fps 13.8 min mp4 | h264 656x480 1.7 Mbps 30 fps 13.7 min mp4 | keep A | keep B | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) |
| e4926beb4d90 | duplicate | h264 640x480 1.2 Mbps 29.97 fps 29.9 min mp4 | wmv3 480x360 0.5 Mbps 29.97 fps 29.9 min wmv | keep B | keep A | Keeps the best copy: more effective detail (640x480 at 1.2 Mbps beats 480x360 at 0.5 Mbps) |
| e4e8aceee29e | duplicate | h264 720x576 1.4 Mbps 25 fps 11.0 min mp4 | h264 720x540 1.9 Mbps 25 fps 11.1 min mp4 | keep A | keep B | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) |
| eb96b1898c90 | duplicate | h264 640x480 1.5 Mbps 30 fps 20.8 min mp4 | wmv2 720x480 2.8 Mbps 23.98 fps 20.8 min wmv | keep B | keep A | Keeps the best copy: same picture at a higher frame rate (30 fps) |
| ef1f5476f39a | duplicate | h264 640x480 1.2 Mbps 29.97 fps 42.2 min mp4 | mpeg4 640x480 2.3 Mbps 29.97 fps 42.3 min avi | keep B | keep A | Keeps the best copy: same picture; the better container |
| f3109b4681da | duplicate | h264 624x480 1.2 Mbps 29.96 fps 19.4 min mp4 | h264 720x540 2.4 Mbps 30 fps 19.1 min mp4 | keep A | keep B | Keeps the best copy: more effective detail (720x540 at 2.4 Mbps beats 624x480 at 1.2 Mbps) |

## Superset (55)

The suggestion keeps the owner's file and one more.

| Record | Kind | Side A | Side B | Human | Suggestion | Reason |
|---|---|---|---|---|---|---|
| 0b6def430012 | duplicate | av1 1440x1080 1.7 Mbps 30 fps 16.1 min mp4 | h264 720x540 1.8 Mbps 30 fps 16.1 min mp4 | keep A | keep B + A | Keeps the best original and the best AI upscale; the other copies are worse. |
| 0b952598847d | duplicate | mpeg4 592x320 1.8 Mbps 29.97 fps 20.3 min avi | h264 720x540 1.4 Mbps 25 fps 17.4 min mp4 | keep A | keep B + A | Keeps the best copy: more effective detail (720x540 at 1.4 Mbps beats 592x320 at 1.8 Mbps) The longest file is kept too because it may hold extra content. |
| 0be85a06ecb0 | duplicate | hevc 1536x1152 9.8 Mbps 29.97 fps 30.1 min mp4 | hevc 1728x1296 19.2 Mbps 29.97 fps 25.2 min mp4 | keep B | keep B + A | Keeps the best copy: more effective detail (1728x1296 at 19.2 Mbps beats 1536x1152 at 9.8 Mbps) The longest file is kept too because it may hold extra content. |
| 10538f776b20 | duplicate | h264 720x540 2.6 Mbps 30 fps 13.1 min mp4 | h264 640x496 1.2 Mbps 24.99 fps 15.2 min mp4 | keep A | keep A + B | Keeps the best copy: more effective detail (720x540 at 2.6 Mbps beats 640x496 at 1.2 Mbps) The longest file is kept too because it may hold extra content. |
| 12c5654da8c7 | duplicate | h264 716x476 0.5 Mbps 29.97 fps 10.9 min mp4 | h264 640x480 0.9 Mbps 29.97 fps 9.3 min mp4 | keep B | keep B + A | Keeps the best copy: more effective detail (640x480 at 0.9 Mbps beats 716x476 at 0.5 Mbps) The longest file is kept too because it may hold extra content. |
| 14e2fd5180fa | duplicate | h264 640x480 1.5 Mbps 30 fps 19.8 min mp4 | wmv2 720x480 2.3 Mbps 23.98 fps 21.8 min wmv | keep B | keep A + B | Keeps the best copy: same picture at a higher frame rate (30 fps) The longest file is kept too because it may hold extra content. |
| 168cb05de85b | duplicate | mpeg4 576x320 1.3 Mbps 29.97 fps 28.3 min avi | av1 1934x1088 10.0 Mbps 29.97 fps 23.8 min mp4 | keep B | keep A + B | Keeps the best original and the best AI upscale; the other copies are worse. |
| 1ac6cef37f2b | duplicate | wmv3 640x480 2.0 Mbps 29.97 fps 19.0 min wmv | mpeg4 576x432 1.2 Mbps 23.98 fps 20.4 min avi | keep A | keep A + B | Keeps the best copy: more effective detail (640x480 at 2.0 Mbps beats 576x432 at 1.2 Mbps) The longest file is kept too because it may hold extra content. |
| 1bba436eae71 | duplicate | av1 1920x1080 7.9 Mbps 29.97 fps 29.8 min mp4 | mpeg4 576x320 1.2 Mbps 29.97 fps 29.8 min avi | keep A | keep B + A | Keeps the best original and the best AI upscale; the other copies are worse. |
| 233eefd96392 | duplicate | av1 1920x1080 10.4 Mbps 29.97 fps 37.4 min mp4 | h264 352x240 0.7 Mbps 29.97 fps 37.2 min mp4 | keep A | keep B + A | Keeps the best original and the best AI upscale; the other copies are worse. |
| 2643cc6266fd | duplicate | h264 646x364 0.5 Mbps 25 fps 24.8 min mp4 | h264 960x540 2.0 Mbps 23.98 fps 23.0 min mp4 | keep B | keep B + A | Keeps the best copy: more effective detail (960x540 at 2.0 Mbps beats 646x364 at 0.5 Mbps) The longest file is kept too because it may hold extra content. |
| 26a21eb40654 | duplicate | av1 956x724 2.9 Mbps 24 fps 62.5 min mp4 | h264 640x482 0.9 Mbps 24 fps 62.8 min mp4 | keep A | keep B + A | Keeps the best original and the best AI upscale; the other copies are worse. |
| 2a18147671b3 | duplicate | h264 720x480 1.3 Mbps 29.97 fps 16.0 min mp4 | mpeg4 480x360 1.6 Mbps 23.98 fps 17.9 min avi | keep A | keep A + B | Keeps the best copy: more effective detail (720x480 at 1.3 Mbps beats 480x360 at 1.6 Mbps) The longest file is kept too because it may hold extra content. |
| 2dfcfdfa1b97 | duplicate | av1 1920x1080 4.4 Mbps 23.98 fps 31.6 min mp4 | h264 640x480 1.3 Mbps 29.97 fps 31.6 min mp4 | keep B | keep B + A | Keeps the best original and the best AI upscale; the other copies are worse. |
| 300627ce29fd | duplicate | h264 720x480 1.3 Mbps 29.97 fps 15.1 min mp4 | h264 656x480 1.9 Mbps 30 fps 13.1 min mp4 | keep A | keep B + A | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) The longest file is kept too because it may hold extra content. |
| 303fcde79ae4 | duplicate | h264 960x544 2.3 Mbps 29.97 fps 28.6 min mp4 | h264 768x432 1.6 Mbps 29.97 fps 32.6 min mp4 | keep B | keep A + B | Keeps the best copy: more effective detail (960x544 at 2.3 Mbps beats 768x432 at 1.6 Mbps) The longest file is kept too because it may hold extra content. |
| 3a368abaaf63 | duplicate | av1 1920x1080 5.4 Mbps 29.97 fps 29.2 min mp4 | h264 640x480 1.2 Mbps 25 fps 29.2 min mp4 | keep A | keep B + A | Keeps the best original and the best AI upscale; the other copies are worse. |
| 3c58133b2a6e | duplicate | h264 720x528 1.3 Mbps 29.97 fps 19.9 min mp4 | h264 720x480 1.5 Mbps 27.25 fps 18.6 min mp4 | keep A | keep B + A | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) The longest file is kept too because it may hold extra content. |
| 3cd87626ff3f | duplicate | h264 1280x720 1.7 Mbps 30 fps 18.6 min mp4 | h264 1920x1440 10.1 Mbps 30 fps 16.8 min mp4 | keep A | keep B + A | Keeps the best copy: more effective detail (1920x1440 at 10.1 Mbps beats 1280x720 at 1.7 Mbps) The longest file is kept too because it may hold extra content. |
| 4281151b7b66 | duplicate | h264 1280x720 3.4 Mbps 29.98 fps 25.3 min mp4 | wmv3 852x480 1.6 Mbps 0 fps 29.4 min wmv | keep A | keep A + B | Keeps the best copy: more effective detail (1280x720 at 3.4 Mbps beats 852x480 at 1.6 Mbps) The longest file is kept too because it may hold extra content. |
| 4437ae16a4b5 | duplicate | av1 1920x1080 5.9 Mbps 29.97 fps 33.7 min mp4 | h264 852x480 1.0 Mbps 30 fps 32.7 min mp4 | keep A | keep B + A | Keeps the best original and the best AI upscale; the other copies are worse. |
| 44451abe6378 | duplicate | h264 720x480 1.5 Mbps 29.97 fps 17.4 min mp4 | h264 720x528 1.3 Mbps 29.97 fps 19.1 min mp4 | keep B | keep A + B | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) The longest file is kept too because it may hold extra content. |
| 46a39544b4c3 | duplicate | h264 720x540 1.9 Mbps 25 fps 13.4 min mp4 | h264 640x496 1.2 Mbps 24.99 fps 15.5 min mp4 | keep A | keep A + B | Keeps the best copy: more effective detail (720x540 at 1.9 Mbps beats 640x496 at 1.2 Mbps) The longest file is kept too because it may hold extra content. |
| 5c1ff1509f2d | duplicate | h264 852x480 1.5 Mbps 30 fps 19.6 min mkv | mpeg4 512x288 1.0 Mbps 23.98 fps 21.0 min avi | keep A | keep A + B | Keeps the best copy: more effective detail (852x480 at 1.5 Mbps beats 512x288 at 1.0 Mbps) The longest file is kept too because it may hold extra content. |
| 5df856912e7f | duplicate | av1 1920x1080 2.8 Mbps 25 fps 40.9 min mp4 | h264 1280x720 3.1 Mbps 30 fps 40.9 min mp4 | keep A | keep B + A | Keeps the best original and the best AI upscale; the other copies are worse. |
| 60462ddd355b | duplicate | av1 1920x1080 4.4 Mbps 23.98 fps 20.4 min mp4 | h264 640x480 1.3 Mbps 29.97 fps 20.4 min mp4 | keep A | keep B + A | Keeps the best original and the best AI upscale; the other copies are worse. |
| 621e932c5570 | duplicate | h264 640x480 1.0 Mbps 25 fps 12.7 min mp4 | av1 3840x2160 10.1 Mbps 50 fps 12.7 min mp4 | keep B | keep A + B | Keeps the best original and the best AI upscale; the other copies are worse. |
| 672ee32d70cf | duplicate | h264 1920x1080 3.1 Mbps 30 fps 26.2 min mp4 | h264 1920x1080 8.2 Mbps 30 fps 24.8 min mp4 | keep A | keep B + A | Keeps the best copy: more effective detail (1920x1080 at 8.2 Mbps beats 1920x1080 at 3.1 Mbps) The longest file is kept too because it may hold extra content. |
| 6a6f041ae18e | duplicate | h264 720x540 1.4 Mbps 25 fps 17.4 min mp4 | mpeg4 576x320 1.9 Mbps 29.97 fps 19.0 min avi | keep B | keep A + B | Keeps the best copy: more effective detail (720x540 at 1.4 Mbps beats 576x320 at 1.9 Mbps) The longest file is kept too because it may hold extra content. |
| 7785e75c60ee | duplicate | av1 1444x1080 2.1 Mbps 60 fps 19.2 min mp4 | mpeg4 352x240 0.9 Mbps 29.97 fps 19.1 min avi | keep A | keep B + A | Keeps the best original and the best AI upscale; the other copies are worse. |
| 78cf88df3407 | duplicate | h264 720x540 1.9 Mbps 25 fps 17.1 min mp4 | h264 640x480 1.2 Mbps 25 fps 18.4 min mp4 | keep A | keep A + B | Keeps the best copy: more effective detail (720x540 at 1.9 Mbps beats 640x480 at 1.2 Mbps) The longest file is kept too because it may hold extra content. |
| 7eb391c890e4 | duplicate | av1 1920x1080 3.8 Mbps 30 fps 36.3 min mp4 | h264 480x360 0.7 Mbps 20 fps 36.3 min mp4 | keep B | keep B + A | Keeps the best original and the best AI upscale; the other copies are worse. |
| 8338e2278936 | duplicate | h264 352x240 0.9 Mbps 29.97 fps 23.6 min mp4 | h264 1280x720 4.0 Mbps 29.97 fps 21.9 min mp4 | keep B | keep B + A | Keeps the best copy: more effective detail (1280x720 at 4.0 Mbps beats 352x240 at 0.9 Mbps) The longest file is kept too because it may hold extra content. |
| 874e2f867294 | duplicate | mpeg4 512x384 1.5 Mbps 29.97 fps 28.3 min avi | wmv1 480x360 0.5 Mbps 0 fps 32.8 min wmv | keep A | keep A + B | Keeps the best copy: more effective detail (512x384 at 1.5 Mbps beats 480x360 at 0.5 Mbps) The longest file is kept too because it may hold extra content. |
| 9686c36840a7 | duplicate | h264 720x540 2.1 Mbps 25 fps 25.2 min mp4 | mpeg4 720x480 2.3 Mbps 29.97 fps 26.7 min avi | keep A | keep A + B | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) The longest file is kept too because it may hold extra content. |
| 96feee8862ef | duplicate | h264 1920x1080 10.2 Mbps 25 fps 2.7 min mp4 | av1 1920x1080 2.2 Mbps 30 fps 8.3 min mp4 | keep B | keep A + B | Keeps the best copy: more effective detail (1920x1080 at 10.2 Mbps beats 1920x1080 at 2.2 Mbps) The longest file is kept too because it may hold extra content. |
| 9a09a9296002 | duplicate | h264 720x540 1.9 Mbps 25 fps 15.7 min mp4 | h264 624x480 1.2 Mbps 29.96 fps 21.1 min mp4 | keep A | keep A + B | Keeps the best copy: more effective detail (720x540 at 1.9 Mbps beats 624x480 at 1.2 Mbps) The longest file is kept too because it may hold extra content. |
| 9affc4abd08e | duplicate | h264 640x480 1.2 Mbps 29.96 fps 20.5 min mp4 | mpeg4 576x432 2.3 Mbps 29.97 fps 21.9 min avi | keep B | keep A + B | Keeps the best copy: more effective detail (640x480 at 1.2 Mbps beats 576x432 at 2.3 Mbps) The longest file is kept too because it may hold extra content. |
| a2c30c373cee | duplicate | h264 656x480 1.9 Mbps 30 fps 13.1 min mp4 | mpeg4 480x360 1.6 Mbps 23.98 fps 16.0 min avi | keep A | keep A + B | Keeps the best copy: more effective detail (656x480 at 1.9 Mbps beats 480x360 at 1.6 Mbps) The longest file is kept too because it may hold extra content. |
| a490d3742343 | duplicate | av1 1920x1080 5.4 Mbps 29.97 fps 31.4 min mp4 | wmv2 480x352 0.9 Mbps 23.98 fps 31.3 min wmv | keep A | keep B + A | Keeps the best original and the best AI upscale; the other copies are worse. |
| aaecbf694b6b | duplicate | av1 1280x960 2.5 Mbps 29.97 fps 28.4 min mp4 | h264 720x480 1.4 Mbps 26.72 fps 30.4 min mp4 | keep A | keep B + A | Keeps the best original and the best AI upscale; the other copies are worse. |
| abf72d1b8c3c | duplicate | h264 720x480 1.3 Mbps 29.97 fps 54.1 min mp4 | mpeg4 512x384 1.5 Mbps 29.97 fps 67.2 min avi | keep A | keep A + B | Keeps the best copy: more effective detail (720x480 at 1.3 Mbps beats 512x384 at 1.5 Mbps) The longest file is kept too because it may hold extra content. |
| ade603b58a39 | duplicate | h264 720x576 1.4 Mbps 25 fps 19.2 min mp4 | h264 720x404 1.9 Mbps 25 fps 20.5 min mp4 | keep B | keep A + B | Keeps the best copy: more effective detail (720x576 at 1.4 Mbps beats 720x404 at 1.9 Mbps) The longest file is kept too because it may hold extra content. |
| b0a48a610e20 | duplicate | h264 512x384 1.1 Mbps 29.97 fps 24.0 min mp4 | av1 1920x1080 5.2 Mbps 29.97 fps 22.5 min mp4 | keep B | keep A + B | Keeps the best original and the best AI upscale; the other copies are worse. |
| b4fa9e49b1af | duplicate | av1 1920x1080 5.4 Mbps 29.97 fps 29.2 min mp4 | h264 640x480 1.6 Mbps 30 fps 29.2 min mp4 | keep A | keep B + A | Keeps the best original and the best AI upscale; the other copies are worse. |
| ba69665cfd72 | duplicate | h264 1920x1080 24.1 Mbps 59.94 fps 29.4 min mp4 | h264 1440x1080 3.1 Mbps 30 fps 33.6 min mp4 | keep A | keep A + B | Keeps the best copy: more effective detail (1920x1080 at 24.1 Mbps beats 1440x1080 at 3.1 Mbps) The longest file is kept too because it may hold extra content. |
| bec980ac1454 | duplicate | h264 1280x720 3.1 Mbps 24 fps 19.3 min mp4 | av1 1920x1080 7.2 Mbps 23.98 fps 19.3 min mp4 | keep B | keep A + B | Keeps the best original and the best AI upscale; the other copies are worse. |
| c18cb1f3c046 | duplicate | av1 1920x1080 2.5 Mbps 30 fps 16.6 min mp4 | h264 1920x1080 1.9 Mbps 29.97 fps 19.8 min mp4 | keep B | keep B + A | Keeps the best original and the best AI upscale; the other copies are worse. |
| c2d194046482 | duplicate | h264 632x480 1.2 Mbps 60 fps 26.1 min mp4 | h264 720x480 2.0 Mbps 59.94 fps 26.1 min mp4 | keep B | keep B + A | Keeps the best original and the best AI upscale; the other copies are worse. |
| c6f32c77cdf3 | duplicate | h264 720x480 1.5 Mbps 29.97 fps 17.8 min mp4 | h264 720x528 1.3 Mbps 29.97 fps 19.4 min mp4 | keep B | keep A + B | Keeps the best copy: more bits per pixel at about the same detail (cleaner, fewer blocks) The longest file is kept too because it may hold extra content. |
| cad5fbdf6e3a | duplicate | h264 720x480 2.6 Mbps 29.97 fps 23.6 min mp4 | av1 1920x1080 5.0 Mbps 29.97 fps 24.0 min mp4 | keep B | keep A + B | Keeps the best original and the best AI upscale; the other copies are worse. |
| cf477b6e633a | duplicate | h264 720x404 1.7 Mbps 23.97 fps 55.6 min mp4 | h264 1920x1080 12.6 Mbps 29.97 fps 50.1 min mp4 | keep B | keep B + A | Keeps the best copy: more effective detail (1920x1080 at 12.6 Mbps beats 720x404 at 1.7 Mbps) The longest file is kept too because it may hold extra content. |
| d72c1185a129 | duplicate | h264 1920x1080 8.0 Mbps 29.97 fps 40.9 min mp4 | h264 576x320 0.8 Mbps 25 fps 45.5 min mp4 | keep A | keep A + B | Keeps the best copy: more effective detail (1920x1080 at 8.0 Mbps beats 576x320 at 0.8 Mbps) The longest file is kept too because it may hold extra content. |
| e4f1eb173982 | duplicate | h264 1280x720 1.4 Mbps 24 fps 20.4 min mp4 | h264 1280x720 1.3 Mbps 30 fps 23.7 min mp4 | keep B | keep A + B | Keeps the best copy: more effective detail (1280x720 at 1.4 Mbps beats 1280x720 at 1.3 Mbps) The longest file is kept too because it may hold extra content. |
| fc5082343b1c | duplicate | h264 1280x720 2.4 Mbps 29.97 fps 10.2 min mp4 | h264 720x480 2.2 Mbps 30 fps 11.8 min mkv | keep A | keep A + B | Keeps the best copy: more effective detail (1280x720 at 2.4 Mbps beats 720x480 at 2.2 Mbps) The longest file is kept too because it may hold extra content. |

## Ties

16 records were practically equal files.

## Deferred

113 contained records are deferred to splitting (#131): the suggestion keeps both files so the long one can be split first, and the owner had deleted the short one.
