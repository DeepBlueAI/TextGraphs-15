AI2 Science Questions Mercury - Without Diagrams v1.1

*****
* Do not distribute *
* Non-commercial use only *
The terms of this data set's license agreement stipulate that this data should not be distributed except by the Allen Institute for Artificial Intelligence (AI2), and only with adherence to the terms contained in the End User License Agreement. All parties interested in acquiring this data must download it from AI2 directly and follow the terms of the the EULA, which specifies the data is to be used for non-commercial, research purposes only.

Please contact ai2-data@allenai.org with any questions regarding AI2’s data sets.
*****


About
——
This data set contains science questions provided under license by a research partner affiliated with AI2. These are English language questions that span several grade levels as indicated in the files. Each question is a 4-way multiple choice structure.
These questions are text-only. They come pre-split into Train, Development, and Test sets. They come in two formats, CSV and JSON. The CSV files contain the full text of the question and its answer options in one cell. The JSON files contain a split version of the question, where the question text has been separated from the answer options programatically.
The question counts are as follows:

Exam01-Elementary-NDMC-Train: 574
Exam01-Elementary-NDMC-Dev: 143
Exam01-Elementary-NDMC-Test: 717

Exam01-MiddleSchool-NDMC-Train: 1582
Exam01-MiddleSchool-NDMC-Dev: 482
Exam01-MiddleSchool-NDMC-Test: 1631


Columns of the CSV
——
questionID: Unique identifier for the question.
originalQuestionID: Legacy ID used within AI2.
totalPossiblePoint: The point value of the question for grading purposes.
AnswerKey: The letter signifying the correct answer option for the question.
isMultipleChoice: 1 indicates the question is multiple choice.
includesDiagram: 0 indicates the question does not include a diagram.
examName: The name of the source exam for these questions.
schoolGrade: The intended grade level for the question.
year: The year the questions were sourced for AI2.
question: The question and its answer options. Each answer option is indicated by a letter in parentheses, e.g., (A) and (B).
subject: The question's subject; this is left blank in this data set.
category: Whether the question is a Train, Dev, or Test question.


Structure of the JSON
——
The JSON files contain the same questions split into the "stem" of the question (the question text) and then the various answer "choices" and their corresponding labels (A, B, C, D). The questionID is also included.

Release notes
__
v. 1.1 (July 2017): Removes 12 extraneous questions (one Middle School NDMC Train, three Middle School NDMC Dev, and eight Middle School NDMC Test)
