##  Kaggle NFL Health & Safety - Helmet Assignment 64th/833

Thanks for the contribution of my teammates [@AlvinAi96](https://github.com/AlvinAi96) and [@Leonrain-Liu](https://github.com/Leonrain-Liu).

### Competition introduction

- competition link: [NFL Health & Safety - Helmet Assignment](https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/overview)
- competition dataset: [Dataset](https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/data)
- competition task: In this competition, you’ll identify and assign football players’  helmets from video footage. In particular, you'll create algorithms  capable of assigning detected helmet impacts to correct players via tracking information. 

### Dependence

```bash
$ pip install easydict
$ sudo apt-get install ffmpeg
```

### Idea and contributions

- A frame edge cropping method to remove the interference of the players and spectators outside the court
- Camera position prediction methods
  - Sideline camera position prediction by rotating  mapping on the original and another-side ngs tracking
  - Endzone camera position prediction by Jersey number plate recognition with OCR
- X/Y axis rotating mapping selection
- Dual deepsort trackers on helmets and players fusion strategy

### Results

- Sideline camera position: 59/60
- Endzone camera position: 51/60
- XY mapping [24 validation videos](https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/discussion/281082): 0.23
- deepsort postprocessing and fusion: 0.529

