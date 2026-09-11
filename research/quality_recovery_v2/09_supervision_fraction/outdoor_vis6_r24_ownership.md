### outdoor_gravel_Tile_0 (radius 24 px)

| scope | n | median | min | Q1-Q3 |
|---|---|---|---|---|
| all views | 151 | 0.994 | 0.537 | 0.944-1.000 |
| pitch_down | 36 | 1.000 | 0.973 | 0.999-1.000 |
| pitch_up | 36 | 0.941 | 0.537 | 0.891-1.000 |
| yaw | 79 | 0.988 | 0.646 | 0.944-0.996 |
| ROI (all) | 63 | 1.000 | 1.000 | 1.000-1.000 |
| ROI pitch_down | 18 | 1.000 | 1.000 | 1.000-1.000 |
| ROI yaw | 45 | 1.000 | 1.000 | 1.000-1.000 |

pixel-weighted supervised share 0.960; views without any LiDAR return: 0 / 151

Tile ownership (returns inside the Tile box + margin; 151 views):

| quantity | n | median | min | Q1-Q3 |
|---|---|---|---|---|
| owned share of returns | 151 | 0.827 | 0.155 | 0.432-0.995 |
| owned share of returns, pitch_down | 36 | 1.000 | 0.884 | 1.000-1.000 |
| owned share of returns, pitch_up | 36 | 0.485 | 0.252 | 0.323-0.972 |
| owned share of returns, yaw | 79 | 0.681 | 0.155 | 0.428-0.879 |
| foreign region share of rgb_mask | 151 | 0.085 | 0.000 | 0.000-0.268 |
| supervised: lidar_support minus foreign region | 151 | 0.861 | 0.313 | 0.636-0.997 |
| supervised: lidar_support minus foreign, pitch_down | 36 | 1.000 | 0.966 | 0.997-1.000 |
| supervised: lidar_support minus foreign, pitch_up | 36 | 0.721 | 0.313 | 0.530-0.999 |
| supervised: lidar_support minus foreign, yaw | 79 | 0.814 | 0.393 | 0.594-0.901 |
| supervised: owned returns only, dilated | 151 | 0.860 | 0.309 | 0.653-0.998 |
| supervised: owned only, pitch_down | 36 | 1.000 | 0.969 | 0.997-1.000 |
| supervised: owned only, pitch_up | 36 | 0.736 | 0.309 | 0.546-1.000 |
| supervised: owned only, yaw | 79 | 0.819 | 0.363 | 0.602-0.913 |
| ROI: lidar_support minus foreign | 63 | 1.000 | 0.910 | 0.980-1.000 |
| ROI: owned returns only | 63 | 1.000 | 0.928 | 0.985-1.000 |
