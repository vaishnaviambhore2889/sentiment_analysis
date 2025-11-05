# 💬 Sentiment Analysis Project
# Created by: Vaishnavi Ambhore & Akanksha Bhadke

# Step 1: Import Libraries
import matplotlib
matplotlib.use('TkAgg')   # for VS Code popup graph

import nltk
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored

# Step 2: Download nltk data (run once)
nltk.download('punkt')
nltk.download('punkt_tab')

# Step 3: Welcome Message
print(colored("\n🌟 SENTIMENT ANALYSIS PROJECT 🌟", "magenta", attrs=["bold"]))
print(colored("Created by: Vaishnavi Ambhore & Akanksha Bhadke", "cyan"))
print("=" * 60)

# Step 4: Take User Input
texts = []
print(colored("\n✏️  Enter your sentences one by one (type 'done' to finish):", "yellow"))

while True:
    user_input = input("👉 Enter a sentence: ")
    if user_input.lower() == "done":
        break
    if user_input.strip() != "":
        texts.append(user_input)

if len(texts) == 0:
    print(colored("\n⚠️  No input provided. Exiting program.", "red"))
    exit()

# Step 5: Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    clean_text = " ".join(words)
    return clean_text

# Step 6: Sentiment Analysis
results = []
print(colored("\n🔍 Analyzing your inputs...\n", "cyan", attrs=["bold"]))

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

# Step 7: Create DataFrame
df = pd.DataFrame(results)

# Step 8: Summary
total = len(df)
positive = len(df[df["Sentiment"] == "Positive"])
negative = len(df[df["Sentiment"] == "Negative"])
neutral = len(df[df["Sentiment"] == "Neutral"])

print("\n" + "=" * 60)
print(colored("📊 SENTIMENT SUMMARY", "blue", attrs=["bold"]))
print("=" * 60)
print(f"✅ Positive: {positive} ({positive/total*100:.1f}%)")
print(f"❌ Negative: {negative} ({negative/total*100:.1f}%)")
print(f"😐 Neutral : {neutral} ({neutral/total*100:.1f}%)")

# Step 9: Visualization
plt.style.use('seaborn-v0_8-poster')
plt.figure(figsize=(7, 5))
df["Sentiment"].value_counts().plot(
    kind="bar",
    color=["limegreen", "salmon", "gold"],
    edgecolor="black"
)
plt.title("💬 Sentiment Distribution", fontsize=16, fontweight="bold")
plt.suptitle("Created by: Vaishnavi Ambhore & Akanksha Bhadke", fontsize=10, color="gray")
plt.xlabel("Sentiment Type", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
