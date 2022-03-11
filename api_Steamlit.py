import os
import pickle
import shap
import json
import requests
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import hydralit_components as hc
import subprocess
import sys

shap.initjs()

@st.cache
def run_api():
    subprocess.Popen([sys.executable, 'api_Flask.py'])

run_api()

@st.cache
def deserialization():
    my_directory = os.path.dirname(__file__)
    pickle_model_objects_path = os.path.join(my_directory, "interpretation_objects.pkl")
    with open(pickle_model_objects_path, "rb") as handle:
        explainer, features, feature_names = pickle.load(handle)
    return explainer, features, feature_names

# Load shap explainer
explainer, features, feature_names = deserialization()


@st.cache #garder dans le cache
def load_data(path):
    df = pd.read_csv(path)
    return df

# Load data
my_directory = os.path.dirname(__file__)
path = os.path.join(my_directory, "dataframe.csv")
df = load_data(path=path)


@st.cache
def split_data(df, num_rows):
    X = df.iloc[:, 2:]
    y = df["TARGET"]
    ids = df["SK_ID_CURR"]
    _, X_test, _, y_test, _, ids = train_test_split(
        X,
        y,
        ids,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_test = X_test.iloc[:num_rows, ]
    y_test = y_test.iloc[:num_rows, ]
    ids = list(ids[:num_rows, ])
    return X_test, y_test, ids


# Split data
X_test, y_test, ids = split_data(df=df, num_rows=1000)


@st.cache(allow_output_mutation=True)
def model_prediction(input):
    url = "http://127.0.0.1:5000/predict" 
    r = requests.post(url, json=input, timeout=120)
    return r.json()["prediction"],r.json()["probability"]




def main():
    # specify the primary menu definition
    menu_data = [
            {'icon': "far fa-chart-bar", 'label':"Feature Importance"},#no tooltip message
            {'icon': "fas fa-tachometer-alt", 'label':"Data Analysis",'ttip':"I'm the Dashboard tooltip!"},
            {'icon': "far fa-address-book", 'label':"Prediction"}, 
            {'icon': "fas fa-folder-plus",'label':"New client"}
    ]
    # we can override any part of the primary colors of the menu
    #over_theme = {'txc_inactive': '#FFFFFF','menu_background':'red','txc_active':'yellow','option_active':'blue'}
    over_theme = {'txc_inactive': '#FFFFFF'}
    page = hc.option_bar(
            option_definition=menu_data,
            title='Dashboard',
            key='PrimaryOption',
            override_theme=over_theme,
            horizontal_orientation=True
        )
    

    st.sidebar.header("Parameters:")
    df_analysis = df.copy()
    for col in df_analysis.filter(like="DAYS").columns:
        df_analysis[col] = df_analysis[col].apply(lambda x: abs(x / 365))
    df_analysis.columns = df_analysis.columns.str.replace("DAYS", "YEARS")
    df_analysis["TARGET"] = df_analysis["TARGET"].astype(str)
    choice_list = list(df_analysis.iloc[:, 2:].columns)

    if page == "Data Analysis":
        st.title("Data Exploration")
        data_analysis = st.sidebar.radio(
            "Choose a type of analysis:",
            ["Univariate",  "Multivariate"],
            index=0,
        )

        if data_analysis == "Univariate":
            st.header("Univariate Analysis")
            options = st.multiselect(
                "Choose a variable to analyse",
                choice_list,
                ["AMT_INCOME_TOTAL", "AMT_CREDIT", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE", "YEARS_BIRTH"])

            if df_analysis[options].select_dtypes(include=["int64", "float64"]).shape[1] > 0:
                graphic_style = st.sidebar.radio(
                    "Select a graphic style for numerical features",
                    ("Histogram", "Box Plot"),
                    index=0,
                )

            if len(options) > 1:
                col1, col2 = st.columns(2)

            for i in range(len(options)):
                if df_analysis[options[i]].dtype == "object":
                    data = df_analysis.groupby("TARGET")[options[i]].value_counts().reset_index(name="percent")
                    data["percent"] = (data["percent"] / len(df_analysis) * 100).round(1)
                    fig = px.bar(
                        data,
                        x=options[i],
                        y="percent",
                        color="TARGET",
                        #text_auto=True,
                        color_discrete_sequence=px.colors.qualitative.Pastel2,
                    )
                    if len(options) > 1:
                        if i % 2 == 0:
                            col1.plotly_chart(fig, use_container_width=True)
                        else:
                            col2.plotly_chart(fig, use_container_width=True)
                    else:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    if graphic_style == "Box Plot":
                        fig = px.box(
                            df_analysis,
                            labels=options[i],
                            y=options[i],
                            points='suspectedoutliers',
                            color="TARGET",
                            category_orders={"TARGET": ["0", "1"]},
                            color_discrete_sequence=px.colors.qualitative.Pastel2,
                        )
                       
                    else:
                        fig = px.histogram(
                            df_analysis,
                            x=options[i],
                            color="TARGET",
                            category_orders={"TARGET": ["0", "1"]},
                            histnorm="percent",
                            nbins=10,
                            color_discrete_sequence=px.colors.qualitative.Pastel2,
                        )
                        fig.update_layout(bargap=0.1)
                    if len(options) > 1:
                        if i % 2 == 0:
                            col1.plotly_chart(fig, use_container_width=True)
                        else:
                            col2.plotly_chart(fig, use_container_width=True)
                    else:
                        st.plotly_chart(fig, use_container_width=True)

        else:
            num_choice_list = list(df_analysis.iloc[:, 2:].select_dtypes(include=["int64", "float64"]).columns)

            container = st.container()
            options = container.multiselect(
                "Choose several numerical variables to analyse:",
                num_choice_list,
                ["AMT_INCOME_TOTAL", "AMT_CREDIT", "EXT_SOURCE_2", "EXT_SOURCE_3"],
            )
            if len(options) > 0:
                import plotly.io as pio
                pio.templates.default = "none"
                corr = df_analysis[["TARGET"] + options]
                corr["TARGET"] = corr["TARGET"].astype(int)
                corr = corr.corr()
                mask = np.zeros_like(corr)
                mask[np.triu_indices_from(mask)] = True
                fig, ax = plt.subplots()
                sns.heatmap(corr, ax=ax,annot=True, fmt=".2f", mask=mask, center=0, cmap="coolwarm")
                plt.title(f"Heatmap des corrélations linéaires\n")
                st.write(fig)
            else:
                st.warning("Please select at least 1 feature")

    elif page == "Prediction":
        sorted_ids = sorted(ids)
        client_id = st.selectbox(
            "Select a client ID:",
            sorted_ids,
        )
        id_idx = ids.index(client_id)
        client_input = X_test.iloc[[id_idx], :]

        st.title(" ")
        st.header("Make a prediction for client #{}".format(client_id))

        predict_button = st.button("Predict")
        if predict_button:
            client_input_json = json.loads(client_input.to_json())
            pred, proba= model_prediction(client_input_json)
            if pred == 0:
                st.success("Loan granted (refund probability = {}%)".format(proba)) 
            else:
                st.error("Loan not granted (default probability = {}%)".format(proba))
            
            st.expander("Show feature impact:")
            force_plot, ax = plt.subplots()
            force_plot = shap.force_plot(
                base_value=explainer.expected_value[pred],
                shap_values=explainer.shap_values[pred][id_idx],
                features=features[id_idx],
                plot_cmap=["#00e800","#ff2839"],
                feature_names=feature_names,
                matplotlib=True,
                show=False,
                
            )
            st.write(force_plot)

            decision_plot, ax = plt.subplots()
            ax = shap.decision_plot(
                base_value=explainer.expected_value[pred],
                shap_values=explainer.shap_values[pred][id_idx],
                features=features[id_idx],
                feature_names=feature_names,
                link='logit',
            )
            st.pyplot(decision_plot)

        with st.expander("Show client information:"):
            df_client_input = pd.DataFrame(
                client_input.to_numpy(),
                index=["Information"],
                columns=client_input.columns,
            ).astype(str).transpose()
            st.dataframe(df_client_input)

    elif page == "Feature Importance":
        st.title("Feature importance for prediction")
        n_features = st.slider(
            "Select number of features:",
            value=7,
            min_value=5,
            max_value=50,
            step=2
            )
        summary_plot, _ = plt.subplots(2,1)
        plt.subplot(121)
        shap.summary_plot(
            shap_values=explainer.shap_values[1],
            features=features,
            feature_names=feature_names,
            max_display=n_features,
            plot_size= [6,2+(n_features/5)],
            color_bar=False
        )
        plt.title("miss credit",size=20)
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks([])

        plt.subplot(122)
        shap.summary_plot(
            shap_values=explainer.shap_values[0],
            features=features,
            feature_names=feature_names,
            max_display=n_features,
            plot_size= [6,2+(n_features/5)]
        )
        plt.yticks([])
        plt.xticks([])
        plt.title("successfull credit",size=20 )
        plt.xlabel("")
        st.pyplot(summary_plot)

    elif page == "New client":
        client_median = X_test.iloc[[1],:]
        #st.write(client_median)
        # Giving a title 
        st.title('Simulation')
        # creating a form
        my_form=st.form(key='form-1')
        # creating input fields
        client_median['AMT_CREDIT']=my_form.text_input('Loan credit:',"20000")
        client_median['AMT_GOODS_PRICE']=my_form.text_input('Previous credit:',"2200200")
        client_median['AMT_ANNUITY']=my_form.text_input('AMT annuity:',"2000")
        client_median['AMT_INCOME_TOTAL']=my_form.text_input('Income total:',"20002")
        # creating radio button 
        my_form.radio('Gender',('M','F'))
        # creating slider 
        my_form.slider('Age:',18,120,25)

        submit=my_form.form_submit_button('Submit')
        if submit:
            client_input_json = json.loads(client_median.to_json())
            pred, proba= model_prediction(client_input_json)
            if pred == 0:
                st.success("Loan granted (refund probability = {}%)".format(proba)) 
            else:
                st.error("Loan not granted (default probability = {}%)".format(proba))
        
        with st.expander("Show client information:"):
            df_client_input = pd.DataFrame(
                client_median.to_numpy(),
                index=["Information"],
                columns=client_median.columns,
            ).astype(str).transpose()
            st.dataframe(df_client_input)
        
    st.markdown('GitHub repository : [https://github.com/polo1093/P7_openclassroom]'
            '(https://github.com/polo1093/P7_openclassroom)')

if __name__ == "__main__":
    main()
