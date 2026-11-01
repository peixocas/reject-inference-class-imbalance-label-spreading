# Data

The LendingClub data are not included in this repository because of their size.

1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club
2. Place the two files in this folder with these names:
   - `accepted_2007_to_2018Q4.csv`
   - `rejected_2007_to_2018Q4.csv`

If the files are compressed (`.csv.gz`), either decompress them or change the two
`pd.read_csv(...)` paths in the notebook to the `.gz` files; pandas reads them directly.
