% Define unit numbers
unit(one, 1).
unit(two, 2).
unit(three, 3).
unit(four, 4).
unit(five, 5).
unit(six, 6).
unit(seven, 7).
unit(eight, 8).
unit(nine, 9).

% Define special numbers from 10 to 19
special(ten, 10).
special(eleven, 11).
special(twelve, 12).
special(thirteen, 13).
special(fourteen, 14).
special(fifteen, 15).
special(sixteen, 16).
special(seventeen, 17).
special(eighteen, 18).
special(nineteen, 19).

% Define tens
tens(twenty, 20).
tens(thirty, 30).
tens(forty, 40).
tens(fifty, 50).
tens(sixty, 60).
tens(seventy, 70).
tens(eighty, 80).
tens(ninety, 90).

% Define multipliers
multiplier(hundred, 100).
multiplier(thousand, 1000).

% Main entry point
words_to_number(Input, Result) :-
    split_string(Input, " ", "", RawWords),
    maplist(string_lower, RawWords, LowercaseWords),
    maplist(atom_string, AtomWords, LowercaseWords),
    process_words(AtomWords, 0, Result).

% Process list of words recursively
process_words([], Acc, Acc).

process_words([and|Tail], Acc, Result) :-
    process_words(Tail, Acc, Result).

process_words([Word, hundred|Tail], Acc, Result) :-
    value_of(Word, Val),
    NewAcc is Acc + Val * 100,
    process_words(Tail, NewAcc, Result).

process_words([Word, thousand|Tail], Acc, Result) :-
    value_of(Word, Val),
    NewAcc is Acc + Val * 1000,
    process_words(Tail, NewAcc, Result).

process_words([Word|Tail], Acc, Result) :-
    value_of(Word, Val),
    NewAcc is Acc + Val,
    process_words(Tail, NewAcc, Result).

% Find value of a word
value_of(Word, Value) :-
    (   unit(Word, Value)
    ;   special(Word, Value)
    ;   tens(Word, Value)
    ;   multiplier(Word, Value)
    ).

