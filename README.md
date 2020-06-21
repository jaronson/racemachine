# Racemachine

## What it Does

This machine tracks visitors to its exhibit, recording their race and sex. When a unique face enters its range of view, it records this data. It makes a tally on a piece of paper divided into six regions comprising white male, black male, asian male, white female, black female and asian female. Once the paper is full, it halts.

## The Intent

Intentionally or otherwise, algorithms that we use on a daily basis have been encoded carrying the biases of their creators. By 2020, technology has advanced to the point that we can programmatically detect sex, race, age, and, by combining factors, in all likelihood, class.

To what end is this technology being used?

## Building a Racist Machine

This machine does nothing more than recording what it detects, but it very well could. We've left that part out and leave it up to the viewer to imagine what actions a machine like this *could* take. 

## Building

Run `pip install -r requirements.txt`

Download the lfwa database and unzip it into `lfwa`

Download the color feret database and unzip it into `colorferet`

Run `ruby scripts/feret_mutator.rb`

Run `scripts/extract-features`

Run `scripts/train`

## Running

Run `scripts/run`

## Notes

Based in part on https://github.com/wondonghyeon/face-classification
