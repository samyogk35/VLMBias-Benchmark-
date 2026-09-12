# VLMBias inventory (main split, revision 3761f9fd7163577534a7816c8bc1004035f2e2a0)

Total rows: 2784

## rows by topic / question / resolution

|                            |   384 |   768 |   1152 |
|:---------------------------|------:|------:|-------:|
| ('Animals', 'Q1')          |    91 |    91 |     91 |
| ('Animals', 'Q2')          |    91 |    91 |     91 |
| ('Chess Pieces', 'Q1')     |    48 |    48 |     48 |
| ('Chess Pieces', 'Q2')     |    48 |    48 |     48 |
| ('Flags', 'Q1')            |    40 |    40 |     40 |
| ('Flags', 'Q2')            |    40 |    40 |     40 |
| ('Game Boards', 'Q1')      |    28 |    28 |     28 |
| ('Game Boards', 'Q2')      |    28 |    28 |     28 |
| ('Logos', 'Q1')            |    69 |    69 |     69 |
| ('Logos', 'Q2')            |    69 |    69 |     69 |
| ('Optical Illusion', 'Q1') |   132 |   132 |    132 |
| ('Optical Illusion', 'Q2') |   132 |   132 |    132 |
| ('Patterned Grid', 'Q1')   |    56 |    56 |     56 |
| ('Patterned Grid', 'Q2')   |    56 |    56 |     56 |

## distinct images and cases per topic

| topic | rows | image files | cases (resolutions collapsed) | rows per case |
|---|---:|---:|---:|---:|
| Animals | 546 | 273 | 91 | 6.0 |
| Chess Pieces | 288 | 144 | 48 | 6.0 |
| Flags | 240 | 120 | 40 | 6.0 |
| Game Boards | 168 | 84 | 28 | 6.0 |
| Logos | 414 | 207 | 69 | 6.0 |
| Optical Illusion | 792 | 396 | 132 | 6.0 |
| Patterned Grid | 336 | 168 | 56 | 6.0 |
| all | 2784 | 1392 | 464 | 6.0 |

Every case shows up 6 times: 3 resolutions x 2 prompt wordings (Q1/Q2).
So the 2784 rows are really 464 cases, and any statistics need to pick one resolution and one prompt.

## optical illusions: ground truth vs expected bias

| sub_topic                    |   ('No', 'Yes') |   ('Yes', 'No') |
|:-----------------------------|----------------:|----------------:|
| Ebbinghaus illusion          |              72 |              72 |
| Müller-Lyer illusion         |              72 |              72 |
| Poggendorff illusion         |              72 |              72 |
| Ponzo illusion               |              72 |              72 |
| Vertical-Horizontal illusion |              36 |              36 |
| Zöllner illusion             |              72 |              72 |

## other splits

- identification: 1392 rows, topics: Animals, Chess Pieces, Flags, Game Boards, Logos, Optical Illusion, Patterned Grid
- withtitle: 2784 rows, topics: Animals, Chess Pieces, Flags, Game Boards, Logos, Optical Illusion, Patterned Grid
- original: 458 rows, topics: Animals, Chess Pieces, Flags, Game Boards, Logos, Optical Illusion
- remove_background_q1q2: 2784 rows, topics: Animals, Chess Pieces, Flags, Game Boards, Logos, Optical Illusions, Patterned Grid
- remove_background_q3: 1392 rows, topics: Animals, Chess Pieces, Flags, Game Boards, Logos, Optical Illusions, Patterned Grid
