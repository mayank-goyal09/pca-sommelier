# 🍷🔬 PCA SOMMELIER 🔬🍷

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=800&color=F97316&center=true&vCenter=true&multiline=false&width=1000&height=80&lines=AI-Powered+Wine+Chemistry+Analysis;PCA+Dimensionality+Reduction+Expert;Interactive+Wine+Profile+Mapping)](https://git.io/typing-svg)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

[![Live App](https://img.shields.io/badge/🌐_Live_App-PCA_Sommelier-FF6B6B?style=for-the-badge)](https://pca-sommelier-project.streamlit.app/)
[![GitHub Stars](https://img.shields.io/github/stars/mayank-goyal09/pca-sommelier?style=social)](https://github.com/mayank-goyal09/pca-sommelier/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/mayank-goyal09/pca-sommelier?style=social)](https://github.com/mayank-goyal09/pca-sommelier/network)

![Wine Data Science](https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif)

### 🎯 **Decode wine chemistry like a data scientist** using **PCA Machine Learning** 🤖
### 🧪 Chemical Features × Dimensionality Reduction = **Wine Intelligence** 🔮

---

## 🌟 **WHAT IS THIS?** 🌟

<table>
<tr>
<td width="50%">

### 🔮 **The Magic**

This **ML-powered app** performs **Principal Component Analysis (PCA)** on wine chemical properties to:
- 📊 Reduce high-dimensional wine data to 2-3 key components
- 🎨 Visualize complex chemical relationships interactively
- 🧠 Identify patterns in wine quality and characteristics
- 📈 Analyze feature loadings and variance explained

**Think of it as:**
- 🍷 **Brain** = PCA Algorithm
- 📝 **Input** = Wine Chemical Properties
- 🎭 **Output** = Reduced Dimensional Space

</td>
<td width="50%">

### ⚡ **Key Features**

✅ **Interactive PCA analysis**  
✅ **Real-time dimensionality reduction**  
✅ **Beautiful Plotly visualizations**  
✅ **Variance & loadings analysis**  
✅ **Sample dataset included**  
✅ **CSV upload support**  
✅ **Component contribution heatmaps**  
✅ **Production-ready Streamlit app**

</td>
</tr>
</table>

---

## 🛠️ **TECH STACK** 🛠️

![Skills](https://skillicons.dev/icons?i=python,github,vscode,git)

| **Category** | **Technologies** |
|-------------|-----------------|
| 🐍 **Language** | Python 3.8+ |
| 📊 **Data Science** | Pandas, NumPy, Scikit-learn |
| 🎨 **Frontend** | Streamlit |
| 📈 **Visualization** | Plotly, Matplotlib, Seaborn |
| 🧪 **ML Algorithm** | PCA (Principal Component Analysis) |
| 🔧 **Preprocessing** | StandardScaler, Feature Selection |

---

## 📂 **PROJECT STRUCTURE** 📂

```
🍷 pca-sommelier/
│
├── 📁 app.py                        # Streamlit web application (CellarScope PCA Studio)
├── 📁 main.ipynb                    # Jupyter notebook with PCA analysis & EDA
├── 📁 build_wine_pca_dataset.py     # Dataset generator script
├── 📦 requirements.txt              # Python dependencies
├── 💾 wine_pca_model.pkl            # Trained PCA model (pickled)
├── 📊 wine_pca_dataset.csv          # Sample wine dataset
├── 📖 README.md                     # You are here!
├── 📄 LICENSE                       # MIT License
└── 🚫 .gitignore                    # Git ignore rules
```

---

## 🚀 **QUICK START** 🚀

![Getting Started](https://user-images.githubusercontent.com/74038190/212257467-871d32b7-e401-42e8-a166-705f7be0b224.gif)

### **Step 1: Clone the Repository** 📥

```bash
git clone https://github.com/mayank-goyal09/pca-sommelier.git
cd pca-sommelier
```

### **Step 2: Install Dependencies** 📦

```bash
pip install -r requirements.txt
```

### **Step 3: Run the App** 🎯

```bash
streamlit run app.py
```

### **Step 4: Open in Browser** 🌐

The app will automatically open at: **`http://localhost:8501`**

---

## 🎮 **HOW TO USE** 🎮

<table>
<tr>
<td width="50%">

### 🔹 **Simple Mode**

1. Open the app
2. Download the sample wine CSV (or upload your own)
3. Select chemical features to analyze
4. Choose number of PCA components (2-10)
5. View interactive visualizations:
   - 📈 Variance explained (Scree plot)
   - 📊 2D PCA projection scatter plot
   - 🧠 Component loadings heatmap

</td>
<td width="50%">

### 🔹 **Data Scientist Mode** 🤓

1. Upload custom wine dataset
2. Explore feature correlations
3. Analyze explained variance ratios
4. Study principal component loadings
5. Identify key chemical drivers
6. Download PCA-transformed data
7. Export visualizations for reports

</td>
</tr>
</table>

---

## 🧪 **HOW IT WORKS** 🧪

```mermaid
graph LR
    A[Wine Chemical Data] --> B[StandardScaler]
    B --> C[Feature Standardization]
    C --> D[PCA Algorithm]
    D --> E[Dimensionality Reduction]
    E --> F[2D/3D Projection]
    F --> G[Variance Analysis]
    G --> H[Component Loadings]
    H --> I[Interactive Visualization]
    style A fill:#FF6B6B
    style D fill:#4ECDC4
    style I fill:#95E1D3
```

### **Pipeline Breakdown:**

1️⃣ **Data Input** → Upload wine CSV with chemical properties  
2️⃣ **Preprocessing** → StandardScaler normalizes features  
3️⃣ **PCA Transformation** → Reduce dimensions while preserving variance  
4️⃣ **Visualization** → Plot principal components and loadings  
5️⃣ **Analysis** → Interpret variance explained and feature importance

---

## 📊 **VISUALIZATIONS** 📊

![Visualization Demo](https://user-images.githubusercontent.com/74038190/212257454-16e3712e-945a-4ca2-b238-408ad0bf87e6.gif)

### 🎨 **What You'll See:**

| **Visualization** | **Description** |
|------------------|----------------|
| 📈 **Scree Plot** | Variance explained by each principal component |
| 📊 **2D Scatter** | Wine samples projected onto PC1 vs PC2 |
| 🧠 **Loadings Heatmap** | Feature contributions to each component |
| 📉 **Cumulative Variance** | Total variance captured by top N components |

### 🔍 **Key Insights:**

- **PC1** typically captures the most variance (e.g., alcohol content)
- **PC2** often represents acidity or sweetness balance
- **Loadings** show which chemical features dominate each component
- **Clustering** in 2D space reveals wine types or quality groups

---

## 💡 **FEATURES** 💡

### ✨ **What Makes This Special?**

```python
# Feature List
features = {
    "Interactive Sliders": "⚡ Choose 2-10 PCA components",
    "Sample Dataset": "📦 Built-in wine chemistry data",
    "Custom Upload": "📤 Use your own CSV files",
    "Real-time PCA": "🔄 Instant dimensionality reduction",
    "Plotly Charts": "📊 Interactive, zoomable visualizations",
    "Variance Explained": "📈 Scree plots & cumulative curves",
    "Loadings Matrix": "🧠 Feature importance heatmaps",
    "Dark Theme UI": "🎨 Professional glass-morphism design",
    "Production Ready": "🚀 Deployable on Streamlit Cloud",
    "Well Documented": "📖 Clear code & comments"
}
```

---

## 📚 **SKILLS DEMONSTRATED** 📚

![Skills](https://user-images.githubusercontent.com/74038190/212257460-738ff738-247f-4445-a718-cdd0ca76e2db.gif)

- ✅ **Unsupervised Learning**: PCA, Dimensionality Reduction
- ✅ **Data Preprocessing**: StandardScaler, Feature Selection
- ✅ **Data Visualization**: Plotly, Matplotlib, Seaborn
- ✅ **Web Development**: Streamlit App Development
- ✅ **Python Libraries**: Pandas, NumPy, Scikit-learn
- ✅ **Statistical Analysis**: Variance, Covariance, Eigenvectors
- ✅ **Model Serialization**: Pickle for saving PCA models
- ✅ **Git & GitHub**: Version Control & Collaboration

---

## 🔬 **DATASET** 🔬

### 📦 **Included Sample Dataset:**

- **178 wine samples** with **13 chemical features**:
  - 🍷 Alcohol content
  - 🧪 Acidity levels (fixed, volatile, citric)
  - 🍬 Residual sugar
  - 🧂 Chlorides
  - 💨 Sulfur dioxide (free & total)
  - 📏 Density
  - 🎨 pH level
  - 🔬 Sulphates
  - ⭐ Quality ratings

### 📤 **Custom Dataset Requirements:**

- CSV format with header row
- At least 2 numeric columns
- No missing values (or minimal NaNs)
- Recommended: 50+ samples for meaningful PCA

---

## 🎯 **USE CASES** 🎯

### 🍷 **Wine Industry Applications:**

- 🔍 **Quality Prediction**: Identify chemical markers of high-quality wines
- 🎨 **Wine Classification**: Group wines by chemical similarity
- 🧪 **Feature Engineering**: Reduce complexity for downstream ML models
- 📊 **Process Optimization**: Find key chemical parameters to control
- 🌍 **Regional Analysis**: Compare wines from different terroirs

### 📚 **Educational Applications:**

- 🎓 Learning PCA concepts interactively
- 📖 Understanding dimensionality reduction
- 🧮 Exploring variance and covariance
- 🔬 Data visualization best practices

---

## 🔮 **FUTURE ENHANCEMENTS** 🔮

- [ ] **3D PCA Visualization** with rotating scatter plots
- [ ] **t-SNE & UMAP** comparison alongside PCA
- [ ] **Automated Feature Selection** based on loadings
- [ ] **Wine Quality Prediction** using reduced components
- [ ] **Biplot Visualization** (samples + feature vectors)
- [ ] **Export Reports** as PDF/HTML with all charts
- [ ] **Multi-dataset Upload** for comparative analysis
- [ ] **API Endpoints** for programmatic access
- [ ] **Mobile-Responsive Design**
- [ ] **Multi-language Support** (Spanish, French, Italian)

---

## 🤝 **CONTRIBUTING** 🤝

![Contribute](https://user-images.githubusercontent.com/74038190/212257465-7ce8d493-cac5-494e-982a-5a9deb852c4b.gif)

Contributions are **always welcome**! 🎉

1. 🍴 **Fork the Project**
2. 🌱 **Create your Feature Branch** (`git checkout -b feature/AmazingFeature`)
3. 💾 **Commit your Changes** (`git commit -m 'Add some AmazingFeature'`)
4. 📤 **Push to the Branch** (`git push origin feature/AmazingFeature`)
5. 🎁 **Open a Pull Request**

---

## 📝 **LICENSE** 📝

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 👨‍💻 **CONNECT WITH ME** 👨‍💻

[![GitHub](https://img.shields.io/badge/GitHub-mayank--goyal09-181717?logo=github&logoColor=white)](https://github.com/mayank-goyal09)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mayank_Goyal-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mayank-goyal-4b8756363/)
[![Email](https://img.shields.io/badge/Email-itsmaygal09%40gmail.com-D14836?logo=gmail&logoColor=white)](mailto:itsmaygal09@gmail.com)

**Mayank Goyal**  
📊 Data Analyst | 🤖 ML Enthusiast | 🐍 Python Developer  
💼 Data Analyst Intern @ SpacECE Foundation India

---

## ⭐ **SHOW YOUR SUPPORT** ⭐

![Support](https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif)

Give a ⭐️ if this project helped you understand PCA better!

### 🍷 **Built with Data Science & ❤️ by Mayank Goyal** 🍷

**"Reducing dimensions, revealing insights, one principal component at a time!"** 🎭

---

[![Portfolio](https://img.shields.io/badge/🌐_Portfolio-Visit_My_Projects-FF6B6B?style=for-the-badge)](https://github.com/mayank-goyal09)

![Footer Wave](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer)