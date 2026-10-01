# Sign Language Digit Recognition

> **Built by [Abdul Qudoos](https://www.abdul-qudoos.com)**, AI Automation & Forward Deployed Engineer · [More projects](https://www.abdul-qudoos.com/work)

A CNN that classifies sign language hand gestures for the digits 0–9, plus a systematic hyperparameter study.

- **Dataset:** 2,060 RGB images at 64×64 (about 206 per digit), 80/20 train/test split
- **Model:** TensorFlow/Keras CNN with early stopping and learning-rate scheduling
- **Experiments:** 10 configurations varying batch size, learning rate, L1/L2 regularisation, dropout and depth (`assignment_experiments.py`)
- **Best result:** 97.58% test accuracy with the deep architecture
- **Notable finding:** batch size 16 reached 96.37% accuracy, while batch size 128 dropped to 73.37%

The full write-up is in `final_report.md`.

## Run it

```bash
pip install -r requirements.txt
python sign_language_main.py
```

---

## About the author

I'm **Abdul Qudoos**, an AI automation and forward deployed engineer based in Islamabad, Pakistan. I build production AI agents, voice agents, workflow automation, and the full-stack products around them.

- Portfolio: [abdul-qudoos.com](https://www.abdul-qudoos.com)
- Case studies: [abdul-qudoos.com/work](https://www.abdul-qudoos.com/work)
- LinkedIn: [Abdul Qudoos](https://www.linkedin.com/in/abdul-qudoos-9a4640324/)
- Email: abdulqudoos7113@gmail.com
