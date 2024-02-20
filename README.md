## Package detection algorithm
**Authors: Lohachov, Tanaka**

This set of algorithms for package detection systems was developed during my internship.
In contrast to the modern ML approaches and intuitive pixel-by-pixel image reconstruction, this algorithm uses the Fourie Transform, which is suitable for drawing borders of repetitive patterns.

### Content:
- `resources` folder contains examples of package photos
- `edge.ipynb` contains algorithms for the image border detection
- `bounds.ipynb` contains algorithms for the individual chip border detection
- `psize.py` is basically a combination of two previous algorithms and only outputs chip size