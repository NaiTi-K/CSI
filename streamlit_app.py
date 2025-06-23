import streamlit as st
import pandas as pd
import pickle
from PIL import Image
with open("tree_model.pkl", "rb") as f:
    model = pickle.load(f)
st.set_page_config(
    page_title="Environmental Risk Dashboard",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
        color: #f0f0f0;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

page = st.sidebar.radio("Choose a Section", [
    "📌 About",
    "🧪 Risk Prediction",
    "📊 Visualizations",
    "🧠 Insights",
    "🌍 Landscape Change",
    "📤 Policy Summary"
])
if page == "📌 About":
    st.title("⚛️ The Price of Power")
    st.markdown("""
    ### India's Nuclear Ambitions and the Cost to Marginalized Communities
    In Jharkhand’s mineral belt, the Uranium Corporation of India Ltd. (UCIL) supports India’s nuclear energy drive. 
    But its legacy has also included environmental degradation, displaced communities, and health crises.

    This project asks:
    > Can a public-sector enterprise like UCIL transform into a model of **ethical mining**, balancing national interest with **environmental protection** and **social justice**?

    **Track:** Tech-AI/ML Track  
    **Focus:** *Data-Driven Environmental Monitoring and Risk Prediction*

    🛰️ We use **remote sensing**, **machine learning**, and **interactive dashboards** to:
    - Detect landscape changes (forest loss, urban expansion, barren land)
    - Predict risk zones based on illnesses and deforestation
    - Support evidence-based policy planning
    """)
elif page == "🧪 Risk Prediction":
    st.title("🧪 Predict Environmental Risk Zones")
    st.markdown("""
    ### Why This?
    UCIL's operations have affected vegetation, air quality, and health in nearby communities.
    This tool helps **simulate regional vulnerability** using:
    - 🌲 Forest loss as ecological degradation
    - 🤒 Reported illnesses as public health markers

    Combined with mining distance, year and zone, a decision tree predicts the **environmental risk level**.
    """)

    with st.expander("ℹ️ About the Inputs"):
        st.markdown("""
        - **Forest Cover Loss (ha):** Deforestation indicator
        - **Reported Illnesses:** Public health indicator
        - **Distance to Mines:** Default 5.0 km for all inputs (proxy exposure)
        - **Year & Zone:** Fixed to allow uniform simulation
        """)

    forest = st.slider("🌲 Forest Cover Loss (ha)", 0.0, 1500.0, 100.0, 10.0)
    illness = st.slider("🤒 Reported Illnesses", 0, 1000, 100, 50)

    with st.expander("📋 Review Input Summary"):
        st.write(f"Forest Loss: {forest} ha")
        st.write(f"Reported Illnesses: {illness}")
        st.write("Distance to Mines: 5.0 km (default)")
        st.write("Year: 2023, Zone: 2 (default)")

    st.info("💡 Use this like a **policy lever** — modify environmental conditions and simulate outcomes.")

    if st.button("🔍 Predict Risk"):
        input_df = pd.DataFrame([{
            'Zone': 2,
            'Year': 2023,
            'Forest_Cover_Loss(ha)': forest,
            'Distance_to_Mines(km)': 5.0,
            'Reported_Illnesses': illness
        }])

        prediction = model.predict(input_df)[0]
        st.markdown("---")

        if prediction == 1:
            st.error("🚨 **High Risk Zone Identified**")
            st.warning("⚠️ Deforestation and health signals indicate ecological stress. Consider mitigation.")
        else:
            st.success("✅ **Low Risk Zone**")
            st.success("🌿 Environmental indicators remain within safe thresholds.")
    st.markdown("---")
    st.subheader("📥 Download Proxy Dataset")
    st.markdown("Use the button below to access the proxy dataset used for model training and interpretation.")
    with open("model\environmental_risk_data.csv", "rb") as file:
        st.download_button("📄 Download CSV", data=file, file_name="environmental_risk_data.csv", mime="text/csv")        
elif page == "📊 Visualizations":
    st.title("📊 Risk Signal Analysis")
    st.markdown("""
    To understand *why* the model makes certain predictions, we visualize the **data distribution**, **feature relevance**, and **class imbalance**.
    """)

    with st.expander("🔗 Correlation Heatmap", expanded=True):
        st.image("heatmap.png", use_container_width=True)
        st.markdown("""
        - Visualizes how features are **related** to each other.
        - 🔍 Helps identify redundancy or inverse signals.
        - Here, forest loss and illnesses show **independent behavior**, which boosts their importance.
        """)

    with st.expander("🌟 Feature Importance", expanded=True):
        st.image("feature_imp.png", use_container_width=True)
        st.markdown("""
        - Decision tree weights shown via bar graph.
        - 🌲 `Forest_Cover_Loss(ha)` and 🤒 `Reported_Illnesses` dominate predictions.
        - Zone, year, distance are nearly irrelevant.
        - ✅ Suggests only 2 features explain most risk — ideal for **rapid field prediction**.
        """)

    with st.expander("📈 Risk Zone Distribution", expanded=True):
        st.image("distribution.png", use_container_width=True)
        st.markdown("""
        - Examines how many zones are **High Risk** vs **Low Risk**.
        - Imbalance matters: too few high-risk zones → model might miss them.
        - Consider **recall/precision tradeoff** in deployment.
        """)

    st.markdown("---")
    st.success("📌 These plots **justify** and **explain** model behavior — crucial for policy usage.")
elif page == "🧠 Insights":
    st.title("🧠 Interpretable AI: How the Model Thinks")
    st.markdown("""
    > Transparency is vital when AI models influence environmental policy. This section shows **why** our model predicts what it does.
    """)

    st.subheader("📉 Confusion Matrix")
    st.image("conf_matrix.png", use_container_width=True)
    st.markdown("""
    - True Positives = High-risk zones correctly flagged
    - False Negatives = Risky zones missed (⚠️ critical to minimize)
    """)

    st.subheader("🌳 Decision Tree")
    st.image("tree.png", use_container_width=True)
    st.markdown("""
    - The tree’s top splits involve `Forest_Cover_Loss` and `Illnesses`
    - Thresholds reveal **decision rules** that are human-readable
    - 🌐 Enables **interpretable risk zones** — not black-box!
    """)

elif page == "🌍 Landscape Change":
    st.title("🌍 Remote Sensing-Based Environmental Change (2013–2023)")
    st.markdown("---")
    st.markdown("""
    This section combines satellite-derived layers to uncover how the landscape has changed over a decade.  
    These insights help us understand the driving factors behind environmental risk in a **spatial and visual** way.
    """)
    st.header("🌿 NDVI Change Detection")
    st.markdown("""
    > To assess **vegetation health and loss**, we use NDVI (Normalized Difference Vegetation Index) — a proxy for green biomass.

    - NDVI was computed using Landsat imagery for the years **2013** and **2023**.
    - By subtracting the 2013 NDVI from 2023 NDVI, we visualize regions that have:
      - **Lost green cover** (deforestation)
      - **Gained vegetation** (reforestation or crop regrowth)
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image("remote_sensing/NDVI_Change_Map.tif", caption="NDVI Change Map (2013–2023)", use_container_width=True)
    with col2:
        st.markdown("### 🌱 NDVI Legend")
        st.markdown("""
        <div style='font-size:14px'>
        <div><span style='color:#FF0000'>■</span> Vegetation Loss (Red)</div>
        <div><span style='color:#0000FF'>■</span> Vegetation Gain (Blue)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    🔍 **Interpretation:**
    - 🔴 Large red patches show widespread vegetation loss — possibly due to mining, urban sprawl, or illegal logging.
    - 🔵 Smaller blue regions show localized regrowth or cropping.
    - This layer gives direct ecological evidence of **where the land has degraded**.
    """)

    st.header("🗽 Land Use Land Cover Classification (LULC)")
    st.markdown("""
    > To understand how land types have evolved, we classified the landscape into:
    - Forest (0), Urban (1), Barren (2), and Water (3)

    Using Random Forest and ESA WorldCover labels, we generated LULC maps for **2013** and **2023**.
    """)

    for year in ["2013", "2023"]:
        st.subheader(f"📍 LULC {year}")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(f"remote_sensing/LULC_{year}_colored.tif", caption=f"LULC {year}", use_container_width=True)
        with col2:
            st.markdown("### 📘 Legend")
            st.markdown("""
            <div style='font-size:14px'>
            <div><span style='color:#00FF00'>■</span> Forest (0)</div>
            <div><span style='color:#FF0000'>■</span> Urban (1)</div>
            <div><span style='color:#D2B48C'>■</span> Barren (2)</div>
            <div><span style='color:#0000FF'>■</span> Water (3)</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    🔍 **Interpretation:**
    - Urban regions (red) expanded considerably between 2013 and 2023.
    - Forest areas (green) were visibly replaced by barren and urban patches.
    - These changes reflect **land degradation** and **habitat conversion**.
    """)
    st.header("🔁 LULC Transition Map")
    st.markdown("""
    > To pinpoint how land types changed (not just where), we computed a **transition map** from the LULC classifications.

    For example:
    - Forest → Urban = 01
    - Urban → Barren = 12
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image("remote_sensing\LULC_Transition_Cleaned_2013_2023.tif", caption="LULC Transitions (2013–2023)", use_container_width=True)
    with col2:
        st.markdown("### 🔄 Transition Legend")
        st.markdown("""
        <div style='font-size:14px'>
        <div><span style='color:#FF0000'>■</span> Forest → Urban (01)</div>
        <div><span style='color:#D2B48C'>■</span> Forest → Barren (02)</div>
        <div><span style='color:#0000FF'>■</span> Forest → Water (03)</div>
        <div><span style='color:#800000'>■</span> Barren → Urban (21)</div>
        <div><span style='color:#B22222'>■</span> Water → Urban (31)</div>
        <div><span style='color:#808080'>■</span> Urban → Barren (12)</div>
        <div><span style='color:#00FF00'>■</span> Urban → Forest (10)</div>
        <div><span style='color:#228B22'>■</span> Water → Forest (30)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    🔍 **Insights:**
    - The most dominant transition is `Forest → Urban` — a sign of urban encroachment.
    - Other worrying transitions include `Forest → Barren`, reflecting deforestation and land degradation.
    - Very few transitions show restoration (`Urban → Forest`) — suggesting **limited recovery**.
    """)

    st.header("🌃 VIIRS Nighttime Lights: Proxy for Urban Growth")
    st.markdown("""
    > VIIRS nightlight data measures emitted brightness from Earth at night — a proxy for human settlement and urban activity.

    We compared composites from **2014** and **2023** to visualize expansion of built-up areas.
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image("remote_sensing/Nightlight_Growth_2014_2023.tif", caption="Nightlight Growth (2014–2023)", use_container_width=True)
    with col2:
        st.markdown("### 💡 Nightlight Legend")
        st.markdown("""
        <div style='font-size:14px'>
        <div><span style='color:red'>■</span> Very Low Brightness</div>
        <div><span style='color:orange'>■</span> Low Brightness</div>
        <div><span style='color:yellow'>■</span> Moderate Brightness</div>
        <div><span style='color:lime'>■</span> Growth Region</div>
        <div><span style='color:cyan'>■</span> High Urban Growth</div>
        <div><span style='color:blue'>■</span> Very Bright Urban</div>
        <div><span style='color:purple'>■</span> City Centers / Peak Urban</div>
        <div><span style='color:black'>■</span> No Change / Masked</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    🔍 **Interpretation:**
    - Urban cores have intensified (blue, purple) and new settlements have emerged (lime, cyan).
    - Spatial patterns match LULC and NDVI transitions — proving **built-up pressure**.
    """)

    st.markdown("## 🔎 Overall Interpretation")
    st.success("""
    ✅ Over 10 years, this region experienced:
    - High vegetation loss and declining NDVI
    - Rapid urban expansion replacing forests and natural land
    - Increased nighttime brightness, confirming built-up growth

    📌 These multi-source insights support **evidence-based planning** for:
    - Conservation zones
    - Urban containment strategies
    - Risk mitigation from ecological degradation
    """)

    



elif page == "📤 Policy Summary":
    st.title("📤 Policy Blueprint Suggestions")
    st.markdown("---")
    st.markdown("""
    Based on our model and visual insights, here’s an interdisciplinary action plan:
    - 🛑 **Regulate zones** with >500 ha forest loss and >70 reported illnesses
    - 🚑 **Deploy mobile healthcare units** to predicted high-risk regions
    - 🛰️ **Integrate satellite monitoring** with local illness data streams
    - 📊 **Publish open dashboards** for mining zone risk
    - 📜 **Update UCIL ESG policy** to account for public health & ecological metrics
    """)

st.markdown("---")
st.markdown("© 2025 • Developed for Tech-AI/ML Track • Environmental Risk Project")
