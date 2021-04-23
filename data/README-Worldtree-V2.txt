Worldtree V2: A Corpus of Science-Domain Structured Explanations
              and Inference Patterns supporting Multi-Hop Inference
Version 2.1 (February 2020)
http://www.cognitiveai.org/explanationbank
=====================================================================

This README describes the usage of the WorldTree corpus of explanations to
standardized elementary and middle school science exam questions (3rd through
9th grade), which also form lexically-connected explanation graphs.  The 
corpus also contains a semistructured Tablestore knowledge resource of 
core elementary/middle school science knowledge as well as common sense/
world knowledge. 

The V2 release of WorldTree includes a large collection of "inference patterns",
or sets of multi-hop patterns over facts that tend to cluster together when
solving questions of particular types (such as questions about changes of state 
of matter, or inherited vs learned characteristics). 

Questions, comments, feedback encouraged: pajansen@email.arizona.edu

CHANGELOG
==========
V1.0:   Initial release with approximately 1700 questions/explanations, and
        5k tablestore facts.  Primarily elementary level (3rd through 5th grade).        

V2.0:   Increased to approximately 4400 questions/explanations (5100 with duplicates
        to study interannotator agreement), and approximately 9k tablestore facts. 
        Extends explanations to include middle school level (6th through 9th grade).
        
V2.1:   Tablestore refactor, now includes 81 tables. Other minor changes.


LICENSE
========
This work combines several datasets, and is distributed under mixed licenses. 
The questions in this corpus are drawn from the AI2 Science Questions V2 corpus,
as well as the separately licensed AI2 Science Questions Mercury dataset
containing science questions provided under license by a research partner 
affiliated with AI2.  A number of the tables in the Tablestore are drawn from 
the AI2 Aristo Tablestore.

For more information, please visit http://www.allenai.org/data.html .


1) AI2 Mercury Dataset
-----------------------------------------
* Do not distribute *
* Non-commercial use only *
The terms of this data set's license agreement stipulate that this data should not 
be distributed except by the Allen Institute for Artificial Intelligence (AI2), and 
only with adherence to the terms contained in the End User License Agreement
(included separately in this archive).
 
All parties interested in acquiring this data must download it from AI2 directly 
and follow the terms of the the EULA, which specifies the data is to be used for 
non-commercial, research purposes only.

Please contact ai2-data@allenai.org with any questions regarding AI2’s data sets.

2) Tablestore, Explanation Annotation, and Inference Patterns
-------------------------------------------------------------
The Tablestore and explanations themselves (separate from the questions) are 
distributed under a CC-BY-SA license.  The Tablestore includes a number of 
tables drawn from the AI2 Aristo Tablestore, which is also distributed under
CC-BY-SA. 

The Creative Commons Attribution-ShareAlike 4.0 International License (http://creativecommons.org/licenses/by-sa/4.0/legalcode)

This means you are free to:
1) Share — copy and redistribute the material in any medium or format
2) Adapt — remix, transform, and build upon the material
for any purpose, even commercially.
The licensor cannot revoke these freedoms as long as you follow the license terms.

Under the following terms:
1) Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
2) ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
3) No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.


ATTRIBUTION
============
If you make use of this resource in your work, please cite the following paper:

Xie, Thiem, Martin, Wainwright, Marmorstein, Jansen (2020). WorldTree V2: A Corpus
of Science-Domain Structured Explanations and Inference Patterns supporting Multi-Hop
Inference (LREC 2020). 

For ease of use, this corpus also includes detailed question classification labels 
for each of the science questions in the ARC corpus.  If you make use of those labels,
please cite the following paper:

Xu, Jansen, Martin, Xie, Yadav, Madabushi, Tafjord, Clark (2020). Multi-class Hierarchical
Question Classification for Multiple Choice Science Exams (LREC 2020). 



USAGE
======
The corpus is distributed in three separate formats: (1) a set of tab-delimited files
describing the questions, explanations, and tablestore, (2) a plain-text version of
the questions and explanations for easy review, (3) a book form, showing the questions
and explanation graphs of the training set. 

* (1) Tab-delimited Questions, Explanations, and Tablestore. 

The primary method of use is through this format, a series of tab-delimited files that
describe the questions, explanations, and tables from which the explanation sentences
are drawn. 

"questions/questions.{train/dev/test}.tsv" are in the AI2 ARC corpus format, and includes
an additional "explanation" field.  This field contains a series of unique identifiers
(UIDs) representing specific table rows, as well as the 'explanatory role' that each 
table row takes on in a given explanation.  Within the explanation field, the UIDs:Role
tuples are pipe-delimited ("|"), and separate UID:Role tuples are space-delimited. 

For example, for the first question:

Q: Which of the following is an example of a form of energy? 
(A) the air in a sealed jar (B) the wire in a metal hanger (C) the water in a small puddle (D) the sound in a loud classroom

The "explanation" field takes the form of four sentences, themselves represented as four 
UIDs referencing specific table rows:

1980-256d-b685-846c|CENTRAL 9f4e-2412-59cc-3383|CENTRAL 7c96-f704-e51e-1678|LEXGLUE 1b3f-b617-d7ef-1815|LEXGLUE

By looking up these rows in the 80+ tables within the tablestore, the plain text 
explanation can be reconstructed: 

Question: Which of the following is an example of a form of energy?    [0]: the air in a sealed jar    [1]: the wire in a metal hanger    [2]: the water in a small puddle    [3]: the sound in a loud classroom    
Correct Answer: 3
Explanation: 
sound is a kind of energy (UID: 1980-256d-b685-846c) (ROLE: CENTRAL)
loud means high in (sound energy ; sound intensity) (UID: 9f4e-2412-59cc-3383) (ROLE: CENTRAL)
form means kind (UID: 7c96-f704-e51e-1678) (ROLE: LEXGLUE)
a kind of something is an example of that something (UID: 1b3f-b617-d7ef-1815) (ROLE: LEXGLUE)


* (2) Plain-text

A plain-text version of the corpus for easy review is also included, in the file
"explanations_plaintext.txt" 


* (3) Book format

*New in WorldTree V2!*

A PDF containing the explanation graphs for the entire training set of the WorldTree V2 corpus
is provided in this archive ("worldtree_explanation_corpus_v2.1_book_desk_reference_draft.pdf"). 
This is intended to be an easy-to-browse human-readable version of the corpus that clearly 
illustrates the large explanation graphs composed by the explanation authors. 


TABLESTORE FORMAT
==================
Location: /tablestore/v2.1/tables/* (tables)
Location: /tablestore/v2.1/tables/tableindex.txt (index of tables)

The first line of each table represents the table header, where each following line 
represents the rows of a given table.  A given column in a table header may be 
preceded by a prefix that marks the column as either filler (e.g. "[FILL]") that
allows the row to be read off as a natural language sentence, or take the form of 
a "[SKIP]" column with meta-data: 

[SKIP] COMMENTS: Annotator comments on this row (if any)
[SKIP] DEP: The tablestore is a living document, and rows are occasionally 
  refactored or moved if they can be better represented in another location. 
  Having text populated in the "DEPrication" column represents that a given
  table row should not actively be used in explanation construction. 
[SKIP] UID: The unique identifier string for a given table row. 



QUESTION/EXPLANATION FILE FORMAT
=================================
Recommended questions: 
Location: /questions/questions.{train/dev/test}.tsv

Raw questions: (not recommended for most uses -- these contain duplicate questions to study interannotator agreement)
Alternate location for raw (unfiltered) questions: /questions/raw/

Field description:
    - `QuestionID`: A unique ID for this question
    - `category`: Which split this question is from (e.g. Train/Test/Dev)
    - `explanation`: List of UUIDs|ROLE tuples that comprise the explanation,
      Separated by spaces.  UUIDs refer to specific rows in the TableStore.
    - `topic`: Detailed question class annotation from Xu et al. (LREC2020). 
       (e.g. MAT_COS == matter change of state)
    - `question`: Question text
    - `flags`: State of the explanation in the annotation pipeline.
       Generally only SUCCESS|READY flagged questions are used.
    - `schoolGrade`: Grade the exam question is given in
    - `AnswerKey`: The correct answer for this question
    - `examName`: Name of exam the question came from


INFERENCE PATTERN FORMAT (GENERAL)
===================================

New in WorldTree V2!  Please see the paper for a detailed explanation.

/inference-patterns/

The pattern data has been broken up into several directories: 
  1) `Pattern tables` The facts involved in the patterns and what roles those facts embody in the pattern.
     /inference-patterns/patterns-tsv/
     
  2) `Edge tables` How the rows in each pattern connect with each other to form a pattern.
     These edges connect the facts of a pattern by shared words. These connections are made automatically 
     in software.
     /inference-patterns/patterns-edges/
     
  3) Human readable versions of the inference patterns
     /inference-patterns/patterns-human-readable-html/    (text descriptions)
     /inference-patterns/patterns-human-readable-svg/     (graphical exports)


INFERENCE PATTERN FORMAT (PATTERN TABLES)
------------------------------------------
Location: /inference-patterns/patterns-tsv/

- The 355 inference patterns generated from the Synchronicity tool
- The files are a standard (\t)tab delimited table of labeled rows in each pattern
- The columns are as follows:
    - `RATING`: Annotator rating for the row
        - `CENTRAL`: Fact is central to the pattern
                              (e.g. Earth's tilt causes the change of the seasons)
        - `CENTRALSW`: Central to the pattern, but slot can be swapped 
                       out with other rows
                       (e.g. Summer/Winter Solstace is on June/December 21st)
        - `GROUNDING`: Fact grounds the pattern to a given situation
                       (e.g. Lemonade is a kind of liquid)
        - `LEXGLUE`: Facts that connect synonymous words
                       (e.g. Daylight means sunlight)
        - ` MAYBE`: Annotator is unsure what role this fact is
        - `BAD`: Fact does not belong to the pattern
        - `UNRATED`: *(Default)* No role assigned to this fact
    - `TABLE`: Name of the table that this row belongs to
    - `ROW`: Row content delimited by columns with " | " (see Tablestore for column definitions)
    - `UID`: UUID for the row in this slot
    - `BACKGROUND_count`: Number of times this row took the role BACKGROUND in training explanations
    - `CENTRAL_count`: Number of times this row took the role CENTRAL training explanations
    - `GROUNDING_count`: Number of times this row took the role GROUNDING training explanations
    - `LEXGLUE_count`: Number of times this row took the role LEXGLUE training explanations
    - `ROLE_count`: Number of times this row took the role ROLE training explanations
    - `lastCount`: Number of times this row was used in other patterns
    - `hintRowUUIDs`: Comma delimited set of rows that also fit in this of the pattern
    - `hintWords`: Annotator hint words for this slot (experimental)
    - `rowNotes`: Annotators notes on the row
    - `OPTIONAL`: Flags whether or not this row is an optional part of the pattern


INFERENCE PATTERN FORMAT (PATTERN EDGES)
-----------------------------------------
Location: /inference-patterns/patterns-edges/

- Edge matrices that define how the facts in each inference pattern interconnect. 
- TO (row) <-> FROM (column) tables that connect each CENTRAL|CENTRALSW|GROUNDING row of the pattern 
  by lexical constraints
- The tables have the following format:
    FROM(col) uuid TO(row) uuid; so it is read column to row
- Each edge(or cell) is formatted as follows:
    - Comma delimiting each constraint
    - Constraints are made from one column of a semi-structured row to another
    - Therefore the column indices are stored, as well as the shared lemma
    - The tables also have a flag and hash for use in the tools that utilizes them.
    - <from_col_idx>:<to_col_idx>:<toggle_edge_flag>:<shared_lemma>:<edge_id>


INFERENCE PATTERN HUMAN-READABLE FORMATS
-----------------------------------------
HTML Location: /inference-patterns/patterns-human-readable-html/

- Minimal interface for viewing the patterns in a table similar to the one 
  used in the tool.
- Standalone files; they do not need the server to be running.


SVG Location: /inference-patterns/patterns-human-readable-svg/

- Graphic export of the patterns as graphs.


