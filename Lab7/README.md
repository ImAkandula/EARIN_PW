Exercise 7 – Logic and Inference (Variant 4)

Task

This program converts numbers written in English words (up to 1000) into their numeric digit form using Prolog.

Example queries:

?- words_to_number("forty five", X).
X = 45.

?- words_to_number("three hundred and sixty seven", X).
X = 367.

?- words_to_number("one thousand", X).
X = 1000.

?- words_to_number("six hundred and ninety nine", X).
X = 699.

How to Run

1. Go to https://swish.swi-prolog.org/
2. Copy and paste the contents of the file lab7.pl.
3. To test the program, run queries like the examples shown above.

Alternatively, if you have SWI-Prolog installed locally:
- Open terminal.
- Run: swipl lab7.pl
- Then run your queries in the Prolog console.

Files Included

- lab7.pl : The Prolog source code implementing Variant 4
- report.md : Report describing logic, components, and challenges
- README.md : Instructions for running the code and example queries

Notes

- The program ignores the optional "and" in number phrases.
- The input is case-insensitive and accepts both lowercase and uppercase words.