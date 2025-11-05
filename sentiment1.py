# 🌟 Sentiment Analysis Project by Vaishnavi Ambhore 🌟

# Step 1: Import Libraries
import matplotlib
matplotlib.use('TkAgg')   # ensures popup window for graph

import nltk
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored

# Step 2: Download Required Data (run once)
nltk.download('punkt')
nltk.download('punkt_tab')

# Step 3: Input Text Data
texts = [
    "I love Python programming!",
    "This movie was terrible and boring.",
    "The weather is nice today.",
    "I'm feeling so happy and excited!",
    "This product is not good at all."
]

# Step 4: Preprocess Function
def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    clean_text = " ".join(words)
    return clean_text

# Step 5: Sentiment Analysis
results = []
print(colored("\n🔍 Analyzing Sentiments...\n", "cyan", attrs=["bold"]))
for text in texts:
    clean = preprocess_text(text)
    blob = TextBlob(clean)
    sentiment = blob.sentiment.polarity

    if sentiment > 0:
        label = "Positive"
        color = "green"
    elif sentiment < 0:
        label = "Negative"
        color = "red"
    else:
        label = "Neutral"
        color = "yellow"

    results.append({"Text": text, "Sentiment": label, "Polarity": sentiment})
    print(f"{colored(label, color, attrs=['bold'])} ➜ {text}")

# Step 6: Convert to DataFrame
df = pd.DataFrame(results)
print("\n" + "="*50)
print(colored("📊 Sentiment Analysis Results", "magenta", attrs=["bold"]))
print("="*50)
print(df)

# Step 7: Summary Statistics
total = len(df)
positive = len(df[df["Sentiment"] == "Positive"])
negative = len(df[df["Sentiment"] == "Negative"])
neutral = len(df[df["Sentiment"] == "Neutral"])

print("\n" + colored("📈 Summary:", "blue", attrs=["bold"]))
print(f"✅ Positive: {positive} ({positive/total*100:.1f}%)")
print(f"❌ Negative: {negative} ({negative/total*100:.1f}%)")
print(f"😐 Neutral : {neutral} ({neutral/total*100:.1f}%)")

# Step 8: Visualization
plt.style.use('seaborn-v0_8-poster')
plt.figure(figsize=(7, 5))
df["Sentiment"].value_counts().plot(
    kind="bar",
    color=["limegreen", "salmon", "gold"],
    edgecolor="black"
)
plt.title("💬 Sentiment Distribution", fontsize=16, fontweight="bold")
plt.suptitle("By Vaishnavi Ambhore", fontsize=10, color="gray")
plt.xlabel("Sentiment Type", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
