# About the sample photos in this folder

The photos in `student_amit_kumar/`, `student_priya_singh/` and
`classroom_test/` are **not** real photographs of real people. They are
grayscale portrait images taken from the **AT&T ("ORL") Database of
Faces**, a long-standing, anonymised academic benchmark dataset created
by AT&T Laboratories Cambridge specifically for face-recognition
research and teaching. Subjects in that dataset are identified only by
anonymous numeric codes (`s1`, `s2`, `s3`, ...), not by name.

For this project they have been:
* relabeled under **fictional student names** (`Amit Kumar`,
  `Priya Singh`) purely so the demo commands read naturally,
* resized and saved as `.jpg` so they work directly with the CLI,
* split into a *training* set (`student_*` folders, used with the
  `register` command) and a held-out *test* set (`classroom_test/`,
  images the model was never trained on, used with the `recognize`
  command) so that the recognition results in the README/report are
  measured on genuinely unseen images.

Reference: F. Samaria and A. Harter, "Parameterisation of a stochastic
model for human face identification", 2nd IEEE Workshop on Applications
of Computer Vision, 1994.

**For your own use of this project**, replace these folders with real
photos of yourself/classmates (with their consent) — see the "Quick
Demo" section of the root `README.md` for the exact folder layout the
`register` command expects.
