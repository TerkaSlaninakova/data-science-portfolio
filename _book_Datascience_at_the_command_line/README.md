# Intro
Great book exposing the practice of working with machine learning workflows and datasets from the command line (rather than jupyter notebook as i'm used to). I refreshed a lot of my linux/bash knowlegde and learned a lot of new stuff / tools / approaches. Highly recommending.

* Book website: http://datascienceatthecommandline.com/
* Book GitHub repository: https://github.com/jeroenjanssens/data-science-at-the-command-line

# Notes
Here are my quick notes from the book together with commands i'd likely forget, but i hope to get back to them. 
Since some chapters went into details on tools i'm not that interested in, i omitted (/scrolled through them), namely: 6. Managing Your Data Workflow and 8. Parallel Pipelines

# Tools for command-line data workflow
* in2csv - xlsx to csv
    - `in2csv data/imdb-250.xlsx > data/imdb-250.csv`
    - `in2csv imdb-250.xlsx | head | cut -c1-80`
    - `in2csv data/imdb-250.xlsx | head | csvcut -c Title,Year,Rating | csvlook`
    - if more sheets, `--sheet <name>`
* csvcut - pipe `cat` of csv file to prettify the table
* sql2csv
* curl
    - `curl -s http://www.gutenberg.org/files/76/76-0.txt | head -n 10`
    - when curling shortened urls, need to add `-L` parameter
    - curl only http headers: `-I`
    - curl an API: `curl -s https://randomuser.me/api | jq .`
        - `jq` for prettifying json output
* curl with oauth - [curlicue](https://github.com/decklin/curlicue)

# Manipulating the data

## Filtering lines (from `seq -f "Line %g" 10 > lines`)
`cat lines `
- first 3
    - ` | head -n 3`
    - ` | sed -n '1,3p'`
    - ` | awk 'NR<=3'`
- last 3
    - ` | head -n 4`
    - ` | sed -n '1,3!p'`
- odd lines
    - ` | sed -n 1~2p`
    - ` | awk 'NR%2'`

## Filtering lines based on a pattern (grep)
- `grep -i chapter alice.txt`

## Replacing values
- `echo 'hello world!' | tr ' ' '_'`

# Exploring data
- TL;DR: `csvstat` is very powerful on its own
```
    --max (maximum)
    --min (minimum)
    --sum (sum)
    --mean (mean)
    --median (median)
    --stdev (standard deviation)
    --nulls (whether column contains nulls)
    --unique (unique values)
    --freq (frequent values)
    --len (max value length)
```
- does the file have header?: `head file.csv | csvlook`
- check out the contents: `less -S file.csv` or `csvlook imdb.csv | less -S`
- print out feature names: `< imdb.csv sed -e 's/,/\n/g;q'`
- select a single column: `cat imdb.csv | csvcut -c Title | tail -n +2` (without header)
- how many unique values are there in a column (can we treat it as identifier?): 
    - `cat imdb.csv | csvcut -c Title | tail -n +2 | sort | uniq | wc -l`
    - `csvstat imdb.csv --unique`

### Example
- get red/white wines dataset
    - `curl -sL http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv > wine-red.csv` ... same for white
- check it out: `head -n 5 wine-{red,white}.csv`
- convert features in both files to lowercase: `for T in red white; do < wine-$T.csv tr '[A-Z]; ' '[a-z],_' | tr -d \" > wine-${T}-clean.csv; done`
- combine the 2 datasets and add new 'type' feature: `csvstack -g red,white -n type wine-{red,white}-clean.csv > wine-both-clean.csv`
- add identifier: `wine-white-clean.csv nl -s, -w1 -v0 | sed '1s/0,/id,/' > wine-white-clean-idd.csv`

## Running experiments
- `run_experiment -l config_file` 
    - requires configuration file:
```
[General]
experiment_name = Wine
task = cross_validate
[Input]
train_location = train
featuresets = [["features.csv"]]
learners = ["LinearRegression","GradientBoostingRegressor","RandomForestRegressor"]
label_col = quality
feature_scaling = both
[Tuning]
grid_search = false
objective = r2
[Output]
log = output
results = output
predictions = output
```

## Jsons
- Get a single feature: `curl -s "https://randomuser.me/api/1.2/?results=5" | jq -r '.results[].email'`

# Feature selection
- balanced shuffled train-test split: `csvstack -n type -g red,white wine-red-clean.csv <(< wine-white-clean.csv ./body.sh shuf | head -n 1600) | csvcut -c fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,ph,sulphates,alcohol,type > wine-balanced.csv`
- check if we have the name number of white vs. red wines: `grep -c white wine-balanced.csv`
- extract header: `< wine-balanced.csv head -n 1 > wine-header.csv`
- split data: `tail -n +2 wine-balanced.csv | shuf | split -d -n r/2`
- join with the header: `cat wine-header.csv x00 > wine-train.csv`, `cat wine-header.csv x01 > wine-test.csv`

## Training with BigML (bigmler)
- obtain API key from https://bigml.com, export BIGML_USERNAME and BIGML_API_KEY in `.bashrc`: `export BIGML_USERNAME=TERKASLANINAKOVA`
- `pip install bigmler`
- `bigmler --train wine-train.csv --test wine-test.csv --prediction-info full --prediction-header --output-dir bigml_output --tag wine --remote`
- cout the number of misclassifications: `paste -d, <(csvcut -c type wine-test.csv) <(csvcut -c type bigml_output/predictions.csv) | awk -F, '{if ($1 != $2) {sum+=1}} END { print sum }'`