# Developed by Hikmet Can Çubukçu

import streamlit as st
st.set_page_config(layout="wide", page_title="APS Simulator", page_icon="🚦")
import statistics
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import time
from math import sqrt
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px


with st.sidebar:
    choice = st.radio("**:blue[Choose the simulation model for setting APS]**" , 
                      ["Setting APS for measurement uncertainty - Analytical rerun simulation", 
                       "Setting APS for measurement uncertainty - Resampling simulation",
                      "Setting APS for imprecision and bias - Analytical rerun simulation", 
                      "Setting APS for imprecision and bias - Resampling simulation"])
    
    with open('./template/template_data_fasting_glucose.xlsx', "rb") as template_file:
        template_byte = template_file.read()
    # download template excel file
    st.download_button(label="Click to Download Template File",
                        data=template_byte,
                        file_name="template_data_fasting_glucose.xlsx",
                        mime='application/octet-stream')
    
    # Upload file widget
    uploaded_file = st.file_uploader(
        '#### **Upload your .xlsx (Excel) or .csv file:**',
        type=['csv', 'xlsx'],
        accept_multiple_files=False
    )

    # Define a cached function to process file data
    @st.cache_data
    def load_data(file):
        # Load the uploaded file (Excel or CSV)
        try:
            df = pd.read_excel(file)
        except:
            df = pd.read_csv(file, sep=None, engine='python')
        return df

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Load the data using the cached function
        uploaded_data = load_data(uploaded_file)

        # Display the selectbox widget outside the cached function
        analyte_name_box = st.selectbox("**Select the Measurand Name**", tuple(uploaded_data.columns))

        # Process the selected analyte data
        analyte_data = uploaded_data[analyte_name_box].dropna().reset_index(drop=True)
    #-----------------------------------------------------------------------    
    # enter number of decimal places of data
    number_of_decimals = st.number_input('**Enter Number of Decimal Places of The Selected Data**', min_value=0, max_value=12)

    #st.subheader('Enter Number of Clinical Decision Limit(s) Below')
    number_CDL = st.number_input('**Enter Number of Clinical Decision Limit(s) Below**', min_value=1, max_value=7)
    
    #st.subheader('Enter Clinical Decision Limit(s) Below')
    if number_CDL == 1:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below:** Please check final category intervals on "Distribution of data" page',min_value=0.00000 ,format="%.f", key=1)
    elif number_CDL == 2:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below (in ascending order):** Please check final category intervals on "Distribution of data" page',min_value=0.00000 ,format="%.f",key=2)
        cdl_2 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f", key=3)
    elif number_CDL == 3:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below (in ascending order):** Please check final category intervals on "Distribution of data" page',min_value=0.00000 ,format="%.f",key=4)
        cdl_2 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f", key=5)
        cdl_3 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f", key=6)
    elif number_CDL == 4:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below (in ascending order):** Please check final category intervals on "Distribution of data" page',min_value=0.00000 ,format="%.f",key=7)
        cdl_2 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f", key=8)
        cdl_3 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f",key=9)
        cdl_4 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f", key=10)
    elif number_CDL == 5:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below (in ascending order):** Please check final category intervals on "Distribution of data" page',min_value=0.00000 ,format="%.f",key=11)
        cdl_2 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f", key=12)
        cdl_3 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f",key=13)
        cdl_4 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f", key=14)
        cdl_5 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f",key=15)
    elif number_CDL == 6:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below (in ascending order):** Please check final category intervals on "Distribution of data" page',min_value=0.00000 ,format="%.f",key=16)
        cdl_2 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f",key=17)
        cdl_3 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f",key=18)
        cdl_4 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f", key=19)
        cdl_5 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f",key=20)
        cdl_6 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f", key=21)
    elif number_CDL == 7:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below (in ascending order):** Please check final category intervals on "Distribution of data" page',min_value=0.00000 ,format="%.f",key=22)
        cdl_2 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f", key=23)
        cdl_3 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f",key=24)
        cdl_4 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f", key=25)
        cdl_5 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f", key=26)
        cdl_6 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f",key=27)
        cdl_7 = st.number_input(label=" ",label_visibility='collapsed',min_value=0.00000 ,format="%.f", key=28)   
    else:
        st.warning('Enter Clinical Decision Limit(s)')
    
    # Aggreement thresholds
    st.subheader('Enter aggreement thresholds below')
    c1, c2, c3 = st.columns(3)
    with c1:
        min_agg_threshold = st.number_input('Minimum', value = 90)
    with c2:
        des_agg_threshold = st.number_input('Desirable', value = 95)
    with c3:
        opt_agg_threshold = st.number_input('Optimal', value = 99) 
    
    if choice == "Setting APS for measurement uncertainty - Resampling simulation" or choice == "Setting APS for imprecision and bias - Resampling simulation":
        cv_i = st.number_input("**:blue[Enter within-subject biological variation]**",min_value=0.1 ,format="%.f", key=283424323)   
        pass    
    # Action button "Simulate & Calculate"
    analyze_button = st.button('**:green[Simulate & Calculate]**')
    
    #st.markdown('---')
    st.info('*Developed by Hikmet Can Çubukçu, MD, MSc, EuSpLM* <hikmetcancubukcu@gmail.com>')
    
 
st.image('./images/APS Calculator (2).png')
st.markdown('#### **:blue[A Data-Driven Tool for Setting Outcome-Based Analytical Performance Specifications for Imprecision, Bias, and Measurement Uncertainty Using Specific Clinical Requirements and Population Data Based on Different Simulation Scenarios]**')
st.markdown('---')
# TABS
tab1, tab2, tab3, tab4 = st.tabs(["📖 **Instructions**", "📊 **:green[Distribution of data]**", 
                                "🚦 **:blue[APS based on overall agreement]**", "🚥 **:violet[APS based on sublevel aggreement]**"],)
with tab1:
    st.markdown("""
                This web application is designed to help laboratory professinals to determine 
                their analytical performance specifications for relative standard measurement uncertainty, imprecision, and bias based on their intended clinical setting and population of concern.

                **If the simulation process exceeds 30 minutes, the web application may encounter interruptions and restart. 
                For more enduring simulations, it is advisable to opt for the desktop application. 
                To do so, kindly download the executable file provided below:**
                """)
    st.markdown("""            
                #### Instructions
                
                1. Upload your data. (e.g. template.xlsx) 
                Make sure that the first row of the Excel file you upload has measurand names and the other rows have analyte values, like the following example:
                
                  | Glucose | LDL-cholesterol |
                  | ----------- | ----------- |
                  | 100 | 90 |
                  | 120 | 120 |
                  | 90 | 100 |
                  | 170 | 110 |        

                2. Select the measurand name. 
                (e.g., for template.xlsx file, "Fasting Glucose (mg/dL)")
                3. Enter the number of decimal places of the selected data 
                (e.g., for 126, the number of decimal places is 0; for 10.95, the number of decimal places is 2). 
                (e.g., for glucose example in the template.xlsx, the number of decimal places is "0")
                4. Enter the number of clinical decision limits you want to include in the APS determination process 
                (e.g., for glucose example in the template.xlsx file, there are two CDLs (100 mg/dL and 126 mg/dL))
                5. Enter the value(s) of clinical decision limit(s) 
                (e.g., for glucose example in the template.xlsx file, CDLs are 100 mg/dL and 126 mg/dL) 
                
                **Note:** Please check the final category intervals on the "Distribution of data" page. 
                
                When more then one CDL is entered, the highest CDL the user entered is considered within the lower category (Categories would be <1stCDL, 1stCDL-2ndCDL, >2ndCDL). 
                For glucose example, when the 2nd CDL is entered as "125", the number 125 is included in the 100-125 interval, 
                while 126 is considered within the interval of >125 (similar to ≥126) 
                
                When one CDL is entered by user, the CDL is considered within the higher category (Categories would be <CDL, ≥CDL).
                
                6. Enter the agreement thresholds that will be used to determine minimum, desirable, and optimal analytical performance specifications.
                7. Click on the "Simulate & Analyze" button.

                Note: For resampling simulation enter within-subject biological variation value. You can obtain within-subject biological variaiton data from biologicalvariation.eu.
                
                """) 


    st.info("""**For setting APS for measurement uncertainty - analytical rerun simulation; Please cite:** *Çubukçu, H.C., Vanstapel, F., Thelen M., van Schrojenstein Lantman, M., 
    Bernabeu-Andreu, F. A., Mesko Brguljan P., Milinkovic, N., Linko, S., Panteghini, M., Boursier, G. 
    (2023) APS Calculator: A Data-Driven Tool for Setting Outcome-Based Analytical Performance Specifications 
    for Measurement Uncertainty Using Specific Clinical Requirements and Population Data. Clinical Chemistry 
    and Laboratory Medicine (CCLM), (Ahead of a print) doi:10.1515/cclm-2023-0740*""")
    
    st.markdown('---')
    content=""" **Disclaimer:**

    *The APS Simulator is provided solely as a decision support tool for assisting users in 
    setting analytical performance specifications based on indirect outcome. 
    It is important to note that while the tool aims to provide valuable guidance, 
    the ultimate responsibility for setting appropriate analytical performance specifications lies with the user.
    The user should exercise caution and consider relevant regulations, scientific literature, 
    and any other applicable guidelines when determining their analytical performance specifications. 
    It is crucial to ensure compliance with privacy, responsibility, and safety standards specific to the user's jurisdiction and intended application.
    While every effort has been made to ensure the accuracy and reliability of the APS Calculator, 
    it is important to understand that no tool can guarantee absolute accuracy or address all possible scenarios. 
    Therefore, users should use their professional judgment and seek additional expert advice when necessary.
    The APS Calculator does not collect any personally identifiable information (PII) or sensitive data from users. 
    However, it is recommended to exercise general caution when using any online application and take necessary 
    precautions to protect your privacy and data security. By using the APS Calculator, users acknowledge and 
    agree that they assume all risks and responsibilities associated with its use. The continuous availability 
    of the calculator cannot be guaranteed, as it may experience periods of unavailability due to various factors, 
    including technical issues, maintenance, or unforeseen circumstances.The creators, developers, and maintainers 
    of the APS Calculator shall not be held liable for any direct or indirect damages or losses arising from its use
    or resulting from the APS calculator's temporary or permanent unavailability. It is advised to regularly 
    review this disclaimer, as it may be updated to reflect any changes in regulations, best practices, or other relevant factors.*
            """
    st.warning(content, icon="⚠️")
    st.markdown('---')
     
# table maker from list 
def create_table(category_intervals):
    # Create the table header
    table = f"|{':blue[Category Intervals]':<12}|\n"
    table += f"|{'-'*12}|\n"
    
    # Add each category to the table
    for category in category_intervals:
        table += f"|{category:<12}|\n"
    
    return table

# bins format checker function
def check_number(num):
    if isinstance(num, int):
        return num
    elif isinstance(num, float):
        if num.is_integer():
            return int(num)
        else:
            num_str = str(num)
            num_decimals = len(num_str.split('.')[1])
            if num_decimals == 1 and num_str.endswith('0'):
                return int(num)
            else:
                return num
    else:
        return "Input is not a number."
    
# converter function for cutting bins
def add_one(num):
    if isinstance(num, int):
        return num + 1
    elif isinstance(num, float):
        if num.is_integer():
            return int(num) + 1
        else:
            num_str = str(num)
            num_decimals = len(num_str.split('.')[1])
            if num_decimals == 1 and num_str.endswith('0'):
                return int(num) + 1
            else:
                return round(num + 0.1**num_decimals, num_decimals)
    else:
        return "Input is not a number."
          
### Subclass accuracy, specificity, sensitivity functions
# subclass specificity calculator
def subclass_specificity(confusion_matrix, class_id):
    """
    confusion matrix of multi-class classification
    class_id: id of a particular class 
    """
    confusion_matrix = np.float64(confusion_matrix)
    TP = confusion_matrix[class_id,class_id]
    FN = np.sum(confusion_matrix[class_id]) - TP
    FP = np.sum(confusion_matrix[:,class_id]) - TP
    TN = np.sum(confusion_matrix) - TP - FN - FP
    # specificity = 0 if TN == 0
    if TN != 0:
        specificity = TN/(TN+FP)
    else:
        specificity = 0.
    return specificity

# subclass sensitivity calculator
def subclass_sensitivity(confusion_matrix, class_id):
    """
    confusion matrix of multi-class classification
    class_id: id of a particular class 
    """
    confusion_matrix = np.float64(confusion_matrix)
    TP = confusion_matrix[class_id,class_id]
    FN = np.sum(confusion_matrix[class_id]) - TP
    FP = np.sum(confusion_matrix[:,class_id]) - TP
    TN = np.sum(confusion_matrix) - TP - FN - FP
    # sensitivity = 0 if TP == 0
    if TP != 0:
        sensitivity = TP/(TP+FN)
    else:
        sensitivity = 0.
    return sensitivity

# subclass accuracy calculator
def subclass_accuracy(confusion_matrix, class_id):
    """
    confusion matrix of multi-class classification
    class_id: id of a particular class 
    """
    confusion_matrix = np.float64(confusion_matrix)
    TP = confusion_matrix[class_id,class_id]
    FN = np.sum(confusion_matrix[class_id]) - TP
    FP = np.sum(confusion_matrix[:,class_id]) - TP
    TN = np.sum(confusion_matrix) - TP - FN - FP
    # accuracy = 0 if TP+TN == 0
    if TP+TN != 0:
        accuracy = (TP+TN)/(TP+FP+FN+TN)
    else:
        accuracy = 0
    return accuracy

### Overall (micro) accuracy, sensitivity, and specificity functions
# micro overall sensitivity
def calculate_micro_overall_sensitivity(confusion_matrix):
    confusion_matrix = np.float64(confusion_matrix)
    TP = np.sum(np.diag(confusion_matrix))
    FN = np.sum(confusion_matrix) - TP
    micro_overall_sensitivity = TP / (TP + FN)
    return micro_overall_sensitivity

# micro_overall_accuracy
def calculate_micro_overall_accuracy(confusion_matrix):
    confusion_matrix = np.float64(confusion_matrix)
    TP = np.diag(confusion_matrix)
    total = np.sum(confusion_matrix) 
    overall_accuracy = np.sum(TP) / np.sum(total)
    return overall_accuracy

# micro_overall_specificity
def calculate_micro_overall_specificity(confusion_matrix):
    confusion_matrix = np.float64(confusion_matrix)
    TP = np.diag(confusion_matrix)
    FN = np.sum(confusion_matrix, axis=1) - TP
    FP = np.sum(confusion_matrix, axis=0) - TP
    TN = np.sum(confusion_matrix) - TP - FN - FP
    overall_specificity = np.sum(TN) / np.sum(TN + FP)
    return overall_specificity
   
# action after clicking the button "simulate & analyze" 
if analyze_button:
    try:
        with st.spinner('**Please wait... Status:**'):
            placeholder = st.empty()
            placeholder.success('**Data preprocessing**', icon = "📂")
            
            column_name = analyte_name_box
            # Numeric data to categorical data conversion
            if number_CDL == 1:
                bins = [0, cdl_1-0.000001,np.inf]
                names = [f'<{check_number(cdl_1)}', f'≥{check_number(cdl_1)}']
                value = [1, 2]

            elif number_CDL == 2: # MODIFED
                bins = [0, cdl_1-0.000001, cdl_2, np.inf]
                names = [f'<{check_number(cdl_1)}', f'{check_number(cdl_1)}-{check_number(cdl_2)}' ,f'>{check_number(cdl_2)}']
                value = [1, 2, 3]

            elif number_CDL == 3:
                bins = [0, cdl_1-0.000001, cdl_2, cdl_3, np.inf]
                names = [f'<{check_number(cdl_1)}', f'{check_number(cdl_1)}-{check_number(cdl_2)}', f'{check_number(add_one(cdl_2))}-{check_number(cdl_3)}' ,f'>{check_number(cdl_3)}']
                value = [1, 2, 3, 4]

            elif number_CDL == 4:
                bins = [0, cdl_1-0.000001, cdl_2, cdl_3, cdl_4, np.inf]
                names = [f'<{check_number(cdl_1)}', f'{check_number(cdl_1)}-{check_number(cdl_2)}', f'{check_number(add_one(cdl_2))}-{check_number(cdl_3)}' ,f'{check_number(add_one(cdl_3))}-{check_number(cdl_4)}',f'>{check_number(cdl_4)}']
                value = [1, 2, 3, 4, 5]

            elif number_CDL == 5:
                bins = [0, cdl_1-0.000001, cdl_2, cdl_3, cdl_4, cdl_5, np.inf]
                names = [f'<{check_number(cdl_1)}', f'{check_number(cdl_1)}-{check_number(cdl_2)}', f'{check_number(add_one(cdl_2))}-{check_number(cdl_3)}' ,f'{check_number(add_one(cdl_3))}-{check_number(cdl_4)}',f'{check_number(add_one(cdl_4))}-{check_number(cdl_5)}',f'>{check_number(cdl_5)}']
                value = [1, 2, 3, 4, 5, 6]

            elif number_CDL == 6:
                bins = [0, cdl_1-0.000001, cdl_2, cdl_3, cdl_4, cdl_5, cdl_6, np.inf]
                names = [f'<{check_number(cdl_1)}', f'{check_number(cdl_1)}-{check_number(cdl_2)}', f'{check_number(add_one(cdl_2))}-{check_number(cdl_3)}' ,f'{check_number(add_one(cdl_3))}-{check_number(cdl_4)}',f'{check_number(add_one(cdl_4))}-{check_number(cdl_5)}',f'{check_number(add_one(cdl_5))}-{check_number(cdl_6)}',f'>{check_number(cdl_6)}']
                value = [1, 2, 3, 4, 5, 6, 7]

            elif number_CDL == 7:
                bins = [0, cdl_1-0.000001, cdl_2, cdl_3, cdl_4, cdl_5, cdl_6, cdl_7, np.inf]
                names = [f'<{check_number(cdl_1)}', f'{check_number(cdl_1)}-{check_number(cdl_2)}', f'{check_number(add_one(cdl_2))}-{check_number(cdl_3)}' ,f'{check_number(add_one(cdl_3))}-{check_number(cdl_4)}',f'{check_number(add_one(cdl_4))}-{check_number(cdl_5)}',f'{check_number(add_one(cdl_5))}-{check_number(cdl_6)}',f'{check_number(add_one(cdl_6))}-{check_number(cdl_7)}',f'>{check_number(cdl_7)}']
                value = [1, 2, 3, 4, 5, 6, 7, 8]
                
            else:
                st.warning('Enter Clinical Decision Limit(s)')
            
            # Numeric data to categorical data conversion
            cat_analyte= pd.cut(analyte_data, bins, labels=names)
            cat_analyte_df= pd.DataFrame(cat_analyte)

            # concat dfs
            cat_analyte_df = cat_analyte_df.rename(columns={column_name: "Analyte_category"})
            analyte_last_df = pd.concat([analyte_data, cat_analyte_df],axis = 1)

            # Category to number conversion
            category_mapping = dict(zip(names, value))
            analyte_last_df['cat_n'] = analyte_last_df['Analyte_category'].cat.rename_categories(category_mapping) 

            with tab2:
                # histogram of original data
                col1, col2 = st.columns([4,1.2])
                col1.write(" ")
                col1.info('Revise your clinical decision limits if the category intervals are not appropriate', icon = "ℹ️")
                col1.info('Make sure you have entered the number of decimal places of the selected data correctly', icon = "ℹ️")
                col2.write(" ")
                col2.write(create_table(names))
                
                st.write(" ")
                st.markdown('##### **:green[Histogram of the original data]**')

                # Get unique categories
                categories_h = analyte_last_df['Analyte_category'].unique()
                
                # Define sorting function to prioritize '<' and '>' categories and sort others based on numbers before '-'
                def custom_sort_key(cat):
                    if cat.startswith("<"):
                        return (0, cat)  # Sort '<' categories first
                    elif cat.startswith(">"):
                        return (2, cat)  # Sort '>' categories last
                    else:
                        num_part = cat.split('-')[0]  # Extract the numbers before '-'
                        try:
                            num = float(num_part)
                            return (1, num)  # Sort other categories based on the extracted number
                        except ValueError:
                            return (1, cat)  # Sort categories without numbers as the original string
                
                # Sort the categories using the custom_sort_key function
                categories_h = sorted(categories_h, key=custom_sort_key)
                
                # Create the histogram figure
                fig_h = go.Figure()             
                # Iterate over categories and add histogram traces
                for i, category in enumerate(categories_h):
                    data = analyte_last_df[analyte_last_df['Analyte_category'] == category][column_name]
                    fig_h.add_trace(go.Histogram(
                        x=data,
                        name=category,
                        opacity=0.7,
                        autobinx=True
                    ))
                # Customize the layout of the figure
                fig_h.update_layout(
                    xaxis=dict(
                        title=column_name,
                        title_font=dict(
                            size=12
                        ),
                        tickfont=dict(
                            size=11
                        ),
                    ),
                    yaxis=dict(
                        title='Count',
                        title_font=dict(
                            size=12
                        ),
                        tickfont=dict(
                            size=11
                        )
                    ),
                    legend=dict(
                        title='Category Intervals',
                        xanchor='right',  # Position the legend on the right
                        yanchor='top',  # Position the legend on the top
                        x=0.98,  # Adjust the x position of the legend
                        y=0.98,  # Adjust the y position of the legend
                        traceorder='normal',
                        font=dict(
                            size=11
                        )
                    ),
                    margin=dict(
                        t=10,
                        r=10,
                        b=10,
                        l=10
                    ),
                    height=500,
                    width=800, barmode='stack'
                )
                # Show the figure using Streamlit
                st.plotly_chart(fig_h, theme="streamlit", use_container_width=True)
            
            # ------------------------------------------------------------------------------------
            placeholder.success('**Simulation**', icon ="🔄")
            ## SIMULATION CODES
            
            ### Selection of original data and original data category
            od = analyte_last_df[column_name] # original data
            o_cat_n = analyte_last_df['cat_n'] # original data category
                                    
            error_l = [] # MU list imprecision
            error_l_b = [] # bias list
            error_l_t = [] # total MU
            
            accuracy_overall_l = [] # accuracy overall list , agreement
            sub_accuracy_score_l = [] # sub_accuracy score (sub group accuracy)

            sensitivity_overall_l = [] # sensitivity overall list
            sub_sensitivity_l = [] # sub_sensitivity score (sub group sensitivity)

            specificity_overall_l = [] # specificity overall list
            sub_specificity_l = [] # sub_specificity score (sub group specificity)

            n_data = len(od) # sample size
                        
            def calculate_mean_lol(list_of_lists):
                # returns mean of every corresponding indexes of sublists
                list_of_lists = list(map(list, zip(*list_of_lists))) # transpose data
                num_elements = len(list_of_lists[0])
                mean = []
                for i in range(num_elements):
                    column_data = [sublist[i] for sublist in list_of_lists]
                    mean.append(statistics.mean(column_data))
                return mean  

            def calculate_submetrics_means(data):
                # returns mean of columns of 10x2 data
                num_columns = len(data[0])  # Assuming all sublists have the same length
                column_means = []

                for i in range(num_columns):
                    column_data = [row[i] for row in data]
                    column_mean = sum(column_data) / len(column_data)
                    column_means.append(column_mean)

                return column_means   
                       
            # MU simulation
            ### MU simulation
            if choice == "Setting APS for measurement uncertainty - Resampling simulation" or choice == "Setting APS for measurement uncertainty - Analytical rerun simulation":
                for e in np.arange(0,0.332,0.001): # CV constant
                    
                    error_ll = [] # MU list

                    accuracy_overall_ll = [] # accuracy overall list , agreement
                    sub_accuracy_score_ll = [] # sub_accuracy score (sub group accuracy)

                    sensitivity_overall_ll = [] # sensitivity overall list
                    sub_sensitivity_ll = [] # sub_sensitivity score (sub group sensitivity)

                    specificity_overall_ll = [] # specificity overall list
                    sub_specificity_ll = [] # sub_specificity score (sub group specificity)
                    
                    # simulation of 10 different random seed functions
                    for s in np.arange(1, 11, 1):          
                        np.random.seed(s+1234) # seed for reproducible results for simulated mu
                        imprec_data_raw = np.random.normal(0, 1, n_data)
                        imprec_data_raw = pd.Series(imprec_data_raw)
                                    
                        n_cat_n = []
                        o_cat_n = list(o_cat_n)
                        result_t1_cat_n = []
                        
                        urel_sim = e # simulated urel
                        result_t1 = od # original result
                        if choice == "Setting APS for measurement uncertainty - Resampling simulation":
                            y_od = result_t1 * (1 + imprec_data_raw * sqrt(urel_sim**2 + (cv_i/100)**2))
                        else:
                            y_od = result_t1 * (1 + imprec_data_raw * urel_sim)  # simulated result, urel_sim applied as fraction

                        
                        nd = round(y_od, number_of_decimals) # round generated values according to number of decimals of the selected data entered by user
                        
                        # Create a dictionary to map old labels (`names`) to new values (`value`)
                        category_mapping = dict(zip(names, value))
                        
                        # Modify categories using rename_categories
                        nd_cat_n = nd_cat.cat.rename_categories(category_mapping)
                        
                        n_cat_n.append(nd_cat_n)
                        n_cat_n = [item for sublist in n_cat_n for item in sublist]
                        n_cat_n = pd.Series(n_cat_n)
                        n_cat_n = n_cat_n.fillna(1)
                        
                        # urel cleaned original data
                        result_t1 = round(result_t1, number_of_decimals) # round MU cleaned values according to number of decimals of the selected data entered by user
                        result_t1_cat= pd.cut(result_t1, bins, labels=names) # Categorization of the new data
                        result_t1_cat_n_value = result_t1_cat.cat.rename_categories(category_mapping)
                        result_t1_cat_n.append(result_t1_cat_n_value)
                        result_t1_cat_n = [item for sublist in result_t1_cat_n for item in sublist]
                        result_t1_cat_n = pd.Series(result_t1_cat_n)
                        result_t1_cat_n = result_t1_cat_n.fillna(1)
                        
                        error_ll.append(e) # MU rate save
                                    
                        matrix = confusion_matrix(result_t1_cat_n, n_cat_n) # establishing confusion matrix
                                    
                        accuracy_overall_ll.append(calculate_micro_overall_accuracy(matrix)) # overall accuracy save to list
                        sensitivity_overall_ll.append(calculate_micro_overall_sensitivity(matrix)) # overall sensitivity save to list
                        specificity_overall_ll.append(calculate_micro_overall_specificity(matrix)) # overall specificity save to list
                                    
                        sub_accuracy_score_lll = []
                        sub_sensitivity_lll = []
                        sub_specificity_lll = []
                        for i in range (0, len(names)):
                            sub_accuracy_score_lll.append(subclass_accuracy(matrix, i)) # subclass accuracy save to list
                            sub_sensitivity_lll.append(subclass_sensitivity(matrix, i)) # subclass sensitivity save to list
                            sub_specificity_lll.append(subclass_specificity(matrix, i)) # subclass specificity save to list
                        sub_accuracy_score_ll.append(sub_accuracy_score_lll)
                        sub_sensitivity_ll.append(sub_sensitivity_lll)
                        sub_specificity_ll.append(sub_specificity_lll)
                    
                    # mean of 10s of sub metrics simulation
                    sub_accuracy_score_ll = calculate_submetrics_means(sub_accuracy_score_ll)
                    sub_sensitivity_ll = calculate_submetrics_means(sub_sensitivity_ll)
                    sub_specificity_ll = calculate_submetrics_means(sub_specificity_ll)
                    
                    error_l.append(error_ll) # MU save to list
                    # metrics append to lists
                    accuracy_overall_l.append(accuracy_overall_ll)
                    sub_accuracy_score_l.append(sub_accuracy_score_ll)
                    
                    sensitivity_overall_l.append(sensitivity_overall_ll)
                    sub_sensitivity_l.append(sub_sensitivity_ll)

                    specificity_overall_l.append(specificity_overall_ll)
                    sub_specificity_l.append(sub_specificity_ll)
                    
                error_l = calculate_mean_lol(error_l)
                accuracy_overall_l = calculate_mean_lol(accuracy_overall_l)
                sensitivity_overall_l = calculate_mean_lol(sensitivity_overall_l)
                specificity_overall_l = calculate_mean_lol(specificity_overall_l)

                ### Accuracy table
                # sub_accuracy score data frame
                sub_accuracy_df = pd.DataFrame(sub_accuracy_score_l)
                sub_accuracy_df.columns = names
                # list to series conversion
                error_l = pd.Series(error_l) 
                accuracy_overall_l = pd.Series(accuracy_overall_l)
                # tables comprised of all data
                error_kappa_table = pd.concat([error_l, accuracy_overall_l], axis = 1)
                error_kappa_table.columns = ["Measurement Uncertainty", "Agreement"]
                error_kappa_table_2 = pd.concat([error_kappa_table, sub_accuracy_df], axis = 1)     
                # reset index
                error_kappa_table_2 = error_kappa_table_2.reset_index() # reset index
                                
                ### Sensitivity table
                # sub_sensitivity data frame
                sub_sensitivity_df = pd.DataFrame(sub_sensitivity_l)
                sub_sensitivity_df.columns = names
                # list to series conversion
                error_l = pd.Series(error_l) 
                sensitivity_overall_l = pd.Series(sensitivity_overall_l)
                # tables comprised of all sensitivity and MU data
                error_sensitivity_table = pd.concat([error_l, sensitivity_overall_l], axis = 1)
                error_sensitivity_table.columns = ["Measurement Uncertainty", "Sensitivity"]
                error_sensitivity_table_2 = pd.concat([error_sensitivity_table, sub_sensitivity_df], axis = 1)
                # reset index
                error_sensitivity_table_2 = error_sensitivity_table_2.reset_index() # reset index
                
                ### Specificity table
                # sub_specificity data frame
                sub_specificity_df = pd.DataFrame(sub_specificity_l)
                sub_specificity_df.columns = names
                # list to series conversion
                error_l = pd.Series(error_l) 
                specificity_overall_l = pd.Series(specificity_overall_l)
                # tables comprised of all specificity and MU data
                error_specificity_table = pd.concat([error_l, specificity_overall_l], axis = 1)
                error_specificity_table.columns = ["Measurement Uncertainty", "Specificity"]
                error_specificity_table_2 = pd.concat([error_specificity_table, sub_specificity_df], axis = 1)  
                # reset index
                error_specificity_table_2 = error_specificity_table_2.reset_index() # reset index
                
                # ------------------------------------------------------------------------------------
                
                ### Disagreement based calculation for accuracy
                bins_k = [0, (min_agg_threshold/100)-0.0000000001, (des_agg_threshold/100)-0.0000000001, (opt_agg_threshold/100)-0.0000000001, np.inf]
                names_k = [f'<{min_agg_threshold}%',f'≥{min_agg_threshold}%', f'≥{des_agg_threshold}%', f'≥{opt_agg_threshold}%']
                cat_Agree= pd.cut(error_kappa_table_2['Agreement'], bins_k, labels=names_k)
                cat_Agree_df= pd.DataFrame(cat_Agree)                        
                # Concat dfs
                cat_Agree_df = cat_Agree_df.rename(columns={"Agreement": "Agreement Category"})
                # Outcome full
                error_kappa_table_2_v2 = pd.concat([error_kappa_table_2, cat_Agree_df],axis = 1)
                error_kappa_table_3 = error_kappa_table_2_v2
                
                ### Disagreement based calculation for sensitivity
                bins_k = [0, (min_agg_threshold/100)-0.0000000001, (des_agg_threshold/100)-0.0000000001, (opt_agg_threshold/100)-0.0000000001, np.inf]
                names_k = [f'<{min_agg_threshold}%',f'≥{min_agg_threshold}%', f'≥{des_agg_threshold}%', f'≥{opt_agg_threshold}%']
                cat_Agree= pd.cut(error_sensitivity_table_2['Sensitivity'], bins_k, labels=names_k)
                cat_Agree_df= pd.DataFrame(cat_Agree)
                # Concat dfs
                cat_Agree_df = cat_Agree_df.rename(columns={"Sensitivity": "Sensitivity Category"})
                # Outcome full
                error_sensitivity_table_2_v2 = pd.concat([error_sensitivity_table_2, cat_Agree_df],axis = 1)
                error_sensitivity_table_3 = error_sensitivity_table_2_v2
                
                ### Disagreement based calculation for specificity
                bins_k = [0, (min_agg_threshold/100)-0.0000000001, (des_agg_threshold/100)-0.0000000001, (opt_agg_threshold/100)-0.0000000001, np.inf]
                names_k = [f'<{min_agg_threshold}%',f'≥{min_agg_threshold}%', f'≥{des_agg_threshold}%', f'≥{opt_agg_threshold}%']
                cat_Agree= pd.cut(error_specificity_table_2['Specificity'], bins_k, labels=names_k)
                cat_Agree_df= pd.DataFrame(cat_Agree)        
                # Concat dfs
                cat_Agree_df = cat_Agree_df.rename(columns={"Specificity": "Specificity Category"})
                # Outcome full
                error_specificity_table_2_v2 = pd.concat([error_specificity_table_2, cat_Agree_df],axis = 1)
                error_specificity_table_3 = error_specificity_table_2_v2
                
                #-------------------------------------------------------------------------------------------
                placeholder.success('**Visualization and Calculation**',icon="📉")
                ## Visiualisation of Data
                ### Contour Plot
                # Accuracy percentage unit conversion by multiplying 100
                error_kappa_table_2_v3 = error_kappa_table_2_v2[['Measurement Uncertainty','Agreement']].apply(lambda x: x*100) # percentage unit conversion
                error_kappa_table_2_v3 = pd.concat([error_kappa_table_2_v3, error_kappa_table_2_v2[['Agreement Category']]], axis = 1) # concat 
                ### Specificity percentage unit conversion by multiplying 100
                error_specificity_table_3_v3 = error_specificity_table_3[['Measurement Uncertainty','Specificity']].apply(lambda x: x*100) # percentage unit conversion
                error_specificity_table_3_v3 = pd.concat([error_specificity_table_3_v3, error_specificity_table_3[['Specificity Category']]], axis = 1) # concat 
                ### Specificity percentage unit conversion by multiplying 100
                error_sensitivity_table_3_v3 = error_sensitivity_table_3[['Measurement Uncertainty','Sensitivity']].apply(lambda x: x*100) # percentage unit conversion
                error_sensitivity_table_3_v3 = pd.concat([error_sensitivity_table_3_v3, error_sensitivity_table_3[['Sensitivity Category']]], axis = 1) # concat 

                with tab3:
                    ### interactive contour plot
                    st.write("##### **:blue[APS for MU based on overall agreement]**")
                    ###### interactive plot

                    limit_xn = min_agg_threshold - 5
                    limit_xp = 100
                    ylim_v1 = error_kappa_table_2_v3[error_kappa_table_2_v3['Agreement'] >= min_agg_threshold - 5]['Measurement Uncertainty'].max()

                    # min, desirable, optimal MU limits
                    ylim_v2 = round(error_kappa_table_2_v3[error_kappa_table_2_v3['Agreement'] >= min_agg_threshold]['Measurement Uncertainty'].max(), 1)
                    ylim_v3 = round(error_kappa_table_2_v3[error_kappa_table_2_v3['Agreement'] >= des_agg_threshold]['Measurement Uncertainty'].max(), 1)
                    ylim_v4 = round(error_kappa_table_2_v3[error_kappa_table_2_v3['Agreement'] >= opt_agg_threshold]['Measurement Uncertainty'].max(), 1)

                    # Define the colors based on the thresholds
                    category_colors = {
                        f'≥{min_agg_threshold}%': 'rgb(166, 0, 0)',    # Dark Red
                        f'≥{des_agg_threshold}%': 'rgb(0, 176, 0)',    # Green
                        f'≥{opt_agg_threshold}%': 'rgb(0, 0, 255)',    # Blue
                    }

                    # Convert Agreement categories to colors
                    error_kappa_table_2_v3['Color'] = error_kappa_table_2_v3['Agreement Category'].map(category_colors)

                    # Create the scatter plot
                    fig = go.Figure()

                    # Add the scatter plot
                    fig.add_trace(go.Scatter(
                        x=error_kappa_table_2_v3['Agreement'],
                        y=error_kappa_table_2_v3['Measurement Uncertainty'],
                        mode='markers',
                        marker=dict(
                            color=error_kappa_table_2_v3['Color'],
                            size=10,
                            opacity=0.7
                        ),
                        name='Agreement',
                        hovertemplate='Agreement: %{x}%<br>Measurement Uncertainty: %{y}%<br>'
                    ))

                    # Add the horizontal lines for percentile agreement thresholds
                    fig.add_shape(
                        type='line',
                        x0=min_agg_threshold,
                        y0=0,
                        x1=min_agg_threshold,
                        y1=ylim_v1,
                        line=dict(
                            color='rgba(255, 0, 0, 0.5)',  # Transparent Dark Red
                            width=1.5,
                            dash='dash'
                        ),
                        xref='x',
                        yref='y'
                    )
                    fig.add_annotation(
                        x=min_agg_threshold + 0.1,
                        y=ylim_v1- 1,
                        text=f'{min_agg_threshold}% Agreement',
                        showarrow=False,
                        xanchor='left',
                        yanchor='middle',
                        font=dict(
                            color='rgba(255, 0, 0, 0.8)'  # Dark Red
                        ),
                        textangle=270  # Rotate the text by 270 degrees
                    )

                    fig.add_shape(
                        type='line',
                        x0=des_agg_threshold,
                        y0=0,
                        x1=des_agg_threshold,
                        y1=ylim_v1,
                        line=dict(
                            color='rgba(0, 176, 0, 0.5)',  # Transparent Green
                            width=1.5,
                            dash='dash'
                        ),
                        xref='x',
                        yref='y'
                    )
                    fig.add_annotation(
                        x=des_agg_threshold + 0.2,  # Adjusted x-coordinate
                        y=ylim_v1 - 1,              # Adjusted y-coordinate
                        text=f'{des_agg_threshold}% Agreement',
                        showarrow=False,
                        xanchor='left',
                        yanchor='middle',
                        font=dict(
                            color='rgba(0, 176, 0, 0.8)'  # Green
                        ),
                        textangle=270  # Rotate the text by 270 degrees
                    )

                    fig.add_shape(
                        type='line',
                        x0=opt_agg_threshold,
                        y0=0,
                        x1=opt_agg_threshold,
                        y1=ylim_v1,
                        line=dict(
                            color='rgba(0, 0, 255, 0.5)',  # Transparent Blue
                            width=1.5,
                            dash='dash'
                        ),
                        xref='x',
                        yref='y'
                    )
                    fig.add_annotation(
                        x=opt_agg_threshold + 0.2,  # Adjusted x-coordinate
                        y=ylim_v1 - 1,              # Adjusted y-coordinate
                        text=f'{opt_agg_threshold}% Agreement',
                        showarrow=False,
                        xanchor='left',
                        yanchor='middle',
                        font=dict(
                            color='rgba(0, 0, 255, 0.8)'  # Blue
                        ),
                        textangle=270  # Rotate the text by 270 degrees
                    )

                    # Set the x and y axis limits
                    fig.update_xaxes(range=[limit_xn, limit_xp])
                    fig.update_yaxes(range=[0, ylim_v1 + 1])

                    # Create the legend
                    fig.update_layout(
                        legend=dict(
                            title='Metrics',
                            x=0.01,
                            y=0.01,
                            traceorder='normal',
                            font=dict(
                                size=11
                            )
                        )
                    )

                    # Set the axis labels and font styles
                    fig.update_layout(
                        xaxis=dict(
                            title='Agreement (%)',
                            title_font=dict(
                                size=14,
                            ),
                            tickfont=dict(
                                size=12,
                            )
                        ),
                        yaxis=dict(
                            title='Relative Standard Measurement Uncertainty (%)',
                            title_font=dict(
                                size=14,
                            ),
                            tickfont=dict(
                                size=12,
                            )
                        ),
                        margin=dict(
                            t=10,
                            r=10,
                            b=10,
                            l=10
                        ),
                        height=500,
                        width=800,
                        plot_bgcolor='rgba(255, 255, 255, 0.9)',  # Light background color
                        paper_bgcolor='rgba(255, 255, 255, 0.9)'  # Light background color
                    )

                    # Text info of APS
                    col22, col33 = st.columns([4, 1])
                    na_quote_1 = ' '
                    na_quote_2 = ' '
                    # >33% MU is unaccceptable
                    if ylim_v2 > 33:
                        ylim_v2 = "NA"
                        na_quote_1 = "NA: Not available"
                    if ylim_v3 > 33:
                        ylim_v3 = "NA"
                        na_quote_1 = "NA: Not available"
                    if ylim_v4 > 33:
                        ylim_v4 = "NA"
                        na_quote_1 = "NA: Not available"
                    
                    # =0% MU is notobtainable
                    if ylim_v2 == 0 or np.isnan(ylim_v2):
                        ylim_v2 = "NO"
                        na_quote_2 = "NO: Not obtainable"
                    if ylim_v3 == 0 or np.isnan(ylim_v3):
                        ylim_v3 = "NO"
                        na_quote_2 = "NO: Not obtainable"
                    if ylim_v4 == 0 or np.isnan(ylim_v4):
                        ylim_v4 = "NO"
                        na_quote_2 = "NO: Not obtainable"
                    col33.write(" ")
                    col33.markdown(f"""
                                | APS level | MU |
                                | ----------- | ----------- |
                                | :red[Minimum] | {ylim_v2} |
                                | :green[Desirable] | {ylim_v3} |
                                | :blue[Optimal] | {ylim_v4} |
                                """)
                    col33.write(na_quote_1) # >33% MU is unaccceptable
                    col33.write(na_quote_2) # =0% MU is notobtainable
                    col22.plotly_chart(fig, theme="streamlit", use_container_width=True) # show figure 
                    st.markdown("---")
                
                with tab4:
                    st.write('##### **:blue[APS for MU based on sublevel aggreement]**')
                    st.write('  ')
                    for i in names:     
                        st.markdown(f'###### **:blue[APS based on sublevel: {i}]**')
                        level_name = i
                        
                        ### Agreement over-sub
                        cat_Agree= pd.cut(error_kappa_table_2[level_name], bins_k, labels=names_k)
                        cat_Agree_df= pd.DataFrame(cat_Agree)
                        # concat dfs
                        cat_Agree_df = cat_Agree_df.rename(columns={level_name: "Agreement Category"})
                        # Outcome full
                        error_kappa_table_2_v2 = pd.concat([error_kappa_table_2, cat_Agree_df],axis = 1)                
                        # percentage unit conversion by multiplying 100
                        error_kappa_table_2_v3 = error_kappa_table_2_v2[['Measurement Uncertainty',level_name]].apply(lambda x: x*100) # percentage unit conversion
                        error_kappa_table_2_v3 = pd.concat([error_kappa_table_2_v3, error_kappa_table_2_v2[['Agreement Category']]], axis = 1) # concat 
                                            
                        ### Sensitivity over-sub
                        cat_Sens= pd.cut(error_sensitivity_table_2[level_name], bins_k, labels=names_k)
                        cat_Sens_df= pd.DataFrame(cat_Sens)
                        # concat dfs
                        cat_Sens_df = cat_Sens_df.rename(columns={level_name: "Sensitivity Category"})
                        # Outcome full
                        error_sensitivity_table_2_v2 = pd.concat([error_sensitivity_table_2, cat_Sens_df],axis = 1)                
                        # percentage unit conversion by multiplying 100
                        error_sensitivity_table_2_v3 = error_sensitivity_table_2_v2[['Measurement Uncertainty',level_name]].apply(lambda x: x*100) # percentage unit conversion
                        error_sensitivity_table_2_v3 = pd.concat([error_sensitivity_table_2_v3, error_sensitivity_table_2_v2[['Sensitivity Category']]], axis = 1) # concat 

                        ### Specificity over-sub
                        cat_Spec= pd.cut(error_specificity_table_2[level_name], bins_k, labels=names_k)
                        cat_Spec_df= pd.DataFrame(cat_Spec)
                        # concat dfs
                        cat_Spec_df = cat_Spec_df.rename(columns={level_name: "Specificity Category"})
                        # Outcome full
                        error_specificity_table_2_v2 = pd.concat([error_specificity_table_2, cat_Spec_df],axis = 1)                
                        # percentage unit conversion by multiplying 100
                        error_specificity_table_2_v3 = error_specificity_table_2_v2[['Measurement Uncertainty',level_name]].apply(lambda x: x*100) # percentage unit conversion
                        error_specificity_table_2_v3 = pd.concat([error_specificity_table_2_v3, error_specificity_table_2_v2[['Specificity Category']]], axis = 1) # concat 

                        # axis limits
                        limit_xn = min_agg_threshold - 5
                        limit_xp = 100
                        ylim_v1 = error_kappa_table_2_v3[error_kappa_table_2_v3[level_name]>=min_agg_threshold-5]['Measurement Uncertainty'].max()

                        # min, desirable, optimal MU limits
                        ylim_v2 = round(error_kappa_table_2_v3[error_kappa_table_2_v3[level_name]>=min_agg_threshold]['Measurement Uncertainty'].max(), 1)
                        ylim_v3 = round(error_kappa_table_2_v3[error_kappa_table_2_v3[level_name]>=des_agg_threshold]['Measurement Uncertainty'].max(), 1)
                        ylim_v4 = round(error_kappa_table_2_v3[error_kappa_table_2_v3[level_name]>=opt_agg_threshold]['Measurement Uncertainty'].max(), 1)

                        # Define the colors based on the thresholds
                        category_colors = {
                            f'≥{min_agg_threshold}%': 'rgb(166, 0, 0)',    # Dark Red
                            f'≥{des_agg_threshold}%': 'rgb(0, 176, 0)',    # Green
                            f'≥{opt_agg_threshold}%': 'rgb(0, 0, 255)',    # Blue
                        }

                        # Convert Agreement categories to colors
                        error_kappa_table_2_v3['Color'] = error_kappa_table_2_v3['Agreement Category'].map(category_colors)
                        
                        # Create the scatter plot
                        fig = go.Figure()

                        # Add the scatter plot of Agreement (accuracy)
                        fig.add_trace(go.Scatter(
                            x=error_kappa_table_2_v3[level_name],
                            y=error_kappa_table_2_v3['Measurement Uncertainty'],
                            mode='markers',
                            marker=dict(
                                color=error_kappa_table_2_v3['Color'],
                                size=10,
                                opacity=0.7
                            ),
                            name='Agreement',
                            hovertemplate='Agreement: %{x}%<br>Measurement Uncertainty: %{y}%<br>'
                        ))

                        # Add the Specificity scatter plot
                        fig.add_trace(go.Scatter(
                            x=error_specificity_table_2_v3[level_name],
                            y=error_specificity_table_2_v3['Measurement Uncertainty'],
                            mode='markers',
                            marker=dict(
                                symbol='pentagon',
                                color= 'rgb(204, 85, 119)' , 
                                size=7,
                                opacity=0.6
                            ),
                            name='Specificity',
                            hovertemplate='Specificity: %{x}%<br>Measurement Uncertainty: %{y}%<br>'
                        ))

                        # Add the Sensitivity scatter plot 
                        fig.add_trace(go.Scatter(
                            x=error_sensitivity_table_2_v3[level_name],
                            y=error_sensitivity_table_2_v3['Measurement Uncertainty'],
                            mode='markers',
                            marker=dict(
                                symbol='hexagon2',
                                color='rgb(128, 0, 128)',
                                size=7,
                                opacity=0.6
                            ),
                            name='Sensitivity',
                            hovertemplate='Sensitivity: %{x}%<br>Measurement Uncertainty: %{y}%<br>'
                        ))

                        # Add the horizontal lines for percentile agreement thresholds
                        fig.add_shape(
                            type='line',
                            x0=min_agg_threshold,
                            y0=0,
                            x1=min_agg_threshold,
                            y1=ylim_v1,
                            line=dict(
                                color='rgba(255, 0, 0, 0.5)',  # Transparent Dark Red
                                width=1.5,
                                dash='dash'
                            ),
                            xref='x',
                            yref='y'
                        )
                        fig.add_annotation(
                            x=min_agg_threshold + 0.1,
                            y=ylim_v1- 1,
                            text=f'{min_agg_threshold}% Agreement',
                            showarrow=False,
                            xanchor='left',
                            yanchor='middle',
                            font=dict(
                                color='rgba(255, 0, 0, 0.8)'  # Dark Red
                            ),
                            textangle=270  # Rotate the text by 270 degrees
                        )

                        fig.add_shape(
                            type='line',
                            x0=des_agg_threshold,
                            y0=0,
                            x1=des_agg_threshold,
                            y1=ylim_v1,
                            line=dict(
                                color='rgba(0, 176, 0, 0.5)',  # Transparent Green
                                width=1.5,
                                dash='dash'
                            ),
                            xref='x',
                            yref='y'
                        )
                        fig.add_annotation(
                            x=des_agg_threshold + 0.2,  # Adjusted x-coordinate
                            y=ylim_v1 - 1,              # Adjusted y-coordinate
                            text=f'{des_agg_threshold}% Agreement',
                            showarrow=False,
                            xanchor='left',
                            yanchor='middle',
                            font=dict(
                                color='rgba(0, 176, 0, 0.8)'  # Green
                            ),
                            textangle=270  # Rotate the text by 270 degrees
                        )

                        fig.add_shape(
                            type='line',
                            x0=opt_agg_threshold,
                            y0=0,
                            x1=opt_agg_threshold,
                            y1=ylim_v1,
                            line=dict(
                                color='rgba(0, 0, 255, 0.5)',  # Transparent Blue
                                width=1.5,
                                dash='dash'
                            ),
                            xref='x',
                            yref='y'
                        )
                        fig.add_annotation(
                            x=opt_agg_threshold + 0.2,  # Adjusted x-coordinate
                            y=ylim_v1 - 1,              # Adjusted y-coordinate
                            text=f'{opt_agg_threshold}% Agreement',
                            showarrow=False,
                            xanchor='left',
                            yanchor='middle',
                            font=dict(
                                color='rgba(0, 0, 255, 0.8)'  # Blue
                            ),
                            textangle=270  # Rotate the text by 270 degrees
                        )

                        # Set the x and y axis limits
                        fig.update_xaxes(range=[limit_xn, limit_xp])
                        fig.update_yaxes(range=[0, ylim_v1 + 1])

                        # Create the legend
                        fig.update_layout(
                            legend=dict(
                                title='Metrics',
                                x=0.01,
                                y=0.01,
                                traceorder='normal',
                                font=dict(
                                    size=11
                                )
                            )
                        )

                        # Set the axis labels and font styles
                        fig.update_layout(
                            xaxis=dict(
                                title='Metrics (%)',
                                title_font=dict(
                                    size=14,
                                ),
                                tickfont=dict(
                                    size=12,
                                )
                            ),
                            yaxis=dict(
                                title='Relative Standard Measurement Uncertainty (%)',
                                title_font=dict(
                                    size=14,
                                ),
                                tickfont=dict(
                                    size=12,
                                )
                            ),
                            margin=dict(
                                t=10,
                                r=10,
                                b=10,
                                l=10
                            ),
                            height=500,
                            width=800,
                            plot_bgcolor='rgba(255, 255, 255, 0.9)',  # Light background color
                            paper_bgcolor='rgba(255, 255, 255, 0.9)'  # Light background color
                        )

                        # Text info of APS
                        col22, col33 = st.columns([4, 1])
                        na_quote_3 = ' '
                        na_quote_4 = ' '
                        # >33% MU is unaccceptable
                        if ylim_v2 > 33:
                            ylim_v2 = "NA"
                            na_quote_3 = "NA: Not available"
                        if ylim_v3 > 33:
                            ylim_v3 = "NA"
                            na_quote_3 = "NA: Not available"
                        if ylim_v4 > 33:
                            ylim_v4 = "NA"
                            na_quote_3 = "NA: Not available"
                        
                        # =0% MU is notobtainable
                        if ylim_v2 == 0 or np.isnan(ylim_v2):
                            ylim_v2 = "NO"
                            na_quote_4 = "NO: Not obtainable"
                        if ylim_v3 == 0 or np.isnan(ylim_v3):
                            ylim_v3 = "NO"
                            na_quote_4 = "NO: Not obtainable"
                        if ylim_v4 == 0 or np.isnan(ylim_v4):
                            ylim_v4 = "NO"
                            na_quote_4 = "NO: Not obtainable"
                        col33.write(" ")
                        col33.markdown(f"""
                                    | APS level | MU |
                                    | ----------- | ----------- |
                                    | :red[Minimum] | {ylim_v2} |
                                    | :green[Desirable] | {ylim_v3} |
                                    | :blue[Optimal] | {ylim_v4} |            
                                    """)
                        col33.write(na_quote_3) # >33% MU is unaccceptable 
                        col33.write(na_quote_4) # =0% MU is notobtainable
                        col22.plotly_chart(fig, theme="streamlit", use_container_width=True) # show figure 
                        st.markdown("---")
                placeholder.success('**Done**', icon="✅")
                time.sleep(2)
                placeholder.empty()



            else:
                for e in np.arange(0,0.34,0.01): # CV constant - 0,0.332,0.001
                    for f in np.arange(-0.33,0.34,0.01): # bias constant
                        f = round(f,2) # change rounding number according to step number
                    
                        error_ll = [] # MU list
                        error_ll_b = [] # bias list
                        
                        # added lists
                        nd_list = [] # new data
                        kappa_score_l = [] # kappa score list

                        accuracy_overall_ll = [] # accuracy overall list , agreement
                        sub_accuracy_score_ll = [] # sub_accuracy score (sub group accuracy)

                        sensitivity_overall_ll = [] # sensitivity overall list
                        sub_sensitivity_ll = [] # sub_sensitivity score (sub group sensitivity)

                        specificity_overall_ll = [] # specificity overall list
                        sub_specificity_ll = [] # sub_specificity score (sub group specificity)
                        
                        # simulation of 10 different random seed functions
                        for s in np.arange(1, 11, 1):          
                            np.random.seed(s+1234) # seed for reproducible results for simulated mu
                            imprec_data_raw = np.random.normal(0, 1, n_data)
                            imprec_data_raw = pd.Series(imprec_data_raw)
                                             
                            
                            n_cat_n = []
                            o_cat_n = list(o_cat_n)
                            result_t1_cat_n = []
                            result_t1 = od # original result
                            e_CVA= e # random error 
                            
                            if choice == "Setting APS for imprecision and bias - Resampling simulation":
                                y_od = result_t1 * (1 + imprec_data_raw*sqrt(e_CVA**2 + (cv_i/100)**2)) 

                            else:
                                y_od = result_t1 * (1 + imprec_data_raw*e_CVA)  # random error applied

                            nd = y_od + result_t1*f # new data, bias applied

                            nd = round(nd, number_of_decimals) # round generated values according to number of decimals of the selected data entered by user
                            nd_cat= pd.cut(nd, bins, labels=names) # Categorization of the new data
                            
                            category_mapping = dict(zip(names, value))
                            nd_cat_n = nd_cat.cat.rename_categories(category_mapping)
                            
                            n_cat_n.append(nd_cat_n)
                            n_cat_n = [item for sublist in n_cat_n for item in sublist]
                            n_cat_n = pd.Series(n_cat_n)
                            n_cat_n = n_cat_n.fillna(1)

                            
                            
                            # urel cleaned original data
                            result_t1 = round(result_t1, number_of_decimals) # round MU cleaned values according to number of decimals of the selected data entered by user
                            result_t1_cat= pd.cut(result_t1, bins, labels=names) # Categorization of the new data
                            result_t1_cat_n_value = result_t1_cat.cat.rename_categories(category_mapping)
                            
                            result_t1_cat_n.append(result_t1_cat_n_value)
                            result_t1_cat_n = [item for sublist in result_t1_cat_n for item in sublist]
                            result_t1_cat_n = pd.Series(result_t1_cat_n)
                            result_t1_cat_n = result_t1_cat_n.fillna(1)
                            
                            error_ll.append(e) # MU rate save
                            error_ll_b.append(f) # bias rate save
                            
                            
                            matrix = confusion_matrix(result_t1_cat_n, n_cat_n) # establishing confusion matrix
                                        
                            accuracy_overall_ll.append(calculate_micro_overall_accuracy(matrix)) # overall accuracy save to list
                            sensitivity_overall_ll.append(calculate_micro_overall_sensitivity(matrix)) # overall sensitivity save to list
                            specificity_overall_ll.append(calculate_micro_overall_specificity(matrix)) # overall specificity save to list
                                        
                            sub_accuracy_score_lll = []
                            sub_sensitivity_lll = []
                            sub_specificity_lll = []
                            for i in range (0, len(names)):
                                sub_accuracy_score_lll.append(subclass_accuracy(matrix, i)) # subclass accuracy save to list
                                sub_sensitivity_lll.append(subclass_sensitivity(matrix, i)) # subclass sensitivity save to list
                                sub_specificity_lll.append(subclass_specificity(matrix, i)) # subclass specificity save to list
                            sub_accuracy_score_ll.append(sub_accuracy_score_lll)
                            sub_sensitivity_ll.append(sub_sensitivity_lll)
                            sub_specificity_ll.append(sub_specificity_lll)
                        
                        # mean of 10s of sub metrics simulation
                        sub_accuracy_score_ll = calculate_submetrics_means(sub_accuracy_score_ll)
                        sub_sensitivity_ll = calculate_submetrics_means(sub_sensitivity_ll)
                        sub_specificity_ll = calculate_submetrics_means(sub_specificity_ll)
                        
                        error_l.append(error_ll) # MU save to list
                        error_l_b.append(error_ll_b) # bias save to list
                        
                        # metrics append to lists
                        accuracy_overall_l.append(accuracy_overall_ll)
                        sub_accuracy_score_l.append(sub_accuracy_score_ll)
                        
                        sensitivity_overall_l.append(sensitivity_overall_ll)
                        sub_sensitivity_l.append(sub_sensitivity_ll)

                        specificity_overall_l.append(specificity_overall_ll)
                        sub_specificity_l.append(sub_specificity_ll)
                
                # mean of simulation metric results    
                error_l = calculate_mean_lol(error_l)
                error_l_b = calculate_mean_lol(error_l_b)
                accuracy_overall_l = calculate_mean_lol(accuracy_overall_l)
                sensitivity_overall_l = calculate_mean_lol(sensitivity_overall_l)
                specificity_overall_l = calculate_mean_lol(specificity_overall_l)

                
                ### Accuracy table
                # sub_accuracy score data frame
                sub_accuracy_df = pd.DataFrame(sub_accuracy_score_l)
                sub_accuracy_df.columns = names
                # list to series conversion
                error_l = pd.Series(error_l) 
                error_l_b = pd.Series(error_l_b)
                accuracy_overall_l = pd.Series(accuracy_overall_l)
                # tables comprised of all data
                error_kappa_table = pd.concat([error_l, error_l_b, accuracy_overall_l], axis = 1)
                error_kappa_table.columns = ["Imprecision", "Bias", "Agreement"]
                error_kappa_table_2 = pd.concat([error_kappa_table, sub_accuracy_df], axis = 1)     
                # reset index
                error_kappa_table_2 = error_kappa_table_2.reset_index() # reset index
                
                ### Sensitivity table
                # sub_sensitivity data frame
                sub_sensitivity_df = pd.DataFrame(sub_sensitivity_l)
                sub_sensitivity_df.columns = names
                # list to series conversion
                error_l = pd.Series(error_l) 
                sensitivity_overall_l = pd.Series(sensitivity_overall_l)
                # tables comprised of all sensitivity and MU data
                error_sensitivity_table = pd.concat([error_l, error_l_b, sensitivity_overall_l], axis = 1)
                error_sensitivity_table.columns = ["Imprecision", "Bias", "Sensitivity"]
                error_sensitivity_table_2 = pd.concat([error_sensitivity_table, sub_sensitivity_df], axis = 1)
                # reset index
                error_sensitivity_table_2 = error_sensitivity_table_2.reset_index() # reset index
                
                ### Specificity table
                # sub_specificity data frame
                sub_specificity_df = pd.DataFrame(sub_specificity_l)
                sub_specificity_df.columns = names
                # list to series conversion
                error_l = pd.Series(error_l) 
                specificity_overall_l = pd.Series(specificity_overall_l)
                # tables comprised of all specificity and MU data
                error_specificity_table = pd.concat([error_l, error_l_b, specificity_overall_l], axis = 1)
                error_specificity_table.columns = ["Imprecision", "Bias", "Specificity"] 
                error_specificity_table_2 = pd.concat([error_specificity_table, sub_specificity_df], axis = 1)  
                # reset index
                error_specificity_table_2 = error_specificity_table_2.reset_index() # reset index
                
                # ------------------------------------------------------------------------------------
                
                ### Disagreement based calculation for accuracy
                bins_k = [0, (min_agg_threshold/100)-0.0000000001, (des_agg_threshold/100)-0.0000000001, (opt_agg_threshold/100)-0.0000000001, np.inf]
                names_k = [f'<{min_agg_threshold}%',f'≥{min_agg_threshold}%', f'≥{des_agg_threshold}%', f'≥{opt_agg_threshold}%']
                cat_Agree= pd.cut(error_kappa_table_2['Agreement'], bins_k, labels=names_k)
                cat_Agree_df= pd.DataFrame(cat_Agree)                        
                # Concat dfs
                cat_Agree_df = cat_Agree_df.rename(columns={"Agreement": "Agreement Category"})
                # Outcome full
                error_kappa_table_2_v2 = pd.concat([error_kappa_table_2, cat_Agree_df],axis = 1)
                error_kappa_table_3 = error_kappa_table_2_v2
                
                ### Disagreement based calculation for sensitivity
                bins_k = [0, (min_agg_threshold/100)-0.0000000001, (des_agg_threshold/100)-0.0000000001, (opt_agg_threshold/100)-0.0000000001, np.inf]
                names_k = [f'<{min_agg_threshold}%',f'≥{min_agg_threshold}%', f'≥{des_agg_threshold}%', f'≥{opt_agg_threshold}%']
                cat_Agree= pd.cut(error_sensitivity_table_2['Sensitivity'], bins_k, labels=names_k)
                cat_Agree_df= pd.DataFrame(cat_Agree)
                # Concat dfs
                cat_Agree_df = cat_Agree_df.rename(columns={"Sensitivity": "Sensitivity Category"})
                # Outcome full
                error_sensitivity_table_2_v2 = pd.concat([error_sensitivity_table_2, cat_Agree_df],axis = 1)
                error_sensitivity_table_3 = error_sensitivity_table_2_v2
                
                ### Disagreement based calculation for specificity
                bins_k = [0, (min_agg_threshold/100)-0.0000000001, (des_agg_threshold/100)-0.0000000001, (opt_agg_threshold/100)-0.0000000001, np.inf]
                names_k = [f'<{min_agg_threshold}%',f'≥{min_agg_threshold}%', f'≥{des_agg_threshold}%', f'≥{opt_agg_threshold}%']
                cat_Agree= pd.cut(error_specificity_table_2['Specificity'], bins_k, labels=names_k)
                cat_Agree_df= pd.DataFrame(cat_Agree)        
                # Concat dfs
                cat_Agree_df = cat_Agree_df.rename(columns={"Specificity": "Specificity Category"})
                # Outcome full
                error_specificity_table_2_v2 = pd.concat([error_specificity_table_2, cat_Agree_df],axis = 1)
                error_specificity_table_3 = error_specificity_table_2_v2
                
                
                
                #-------------------------------------------------------------------------------------------
                placeholder.success('**Visualization and Calculation**',icon="📉")
                ## Visiualisation of Data
                ### Contour Plot
                # Accuracy percentage unit conversion by multiplying 100
                error_kappa_table_2_v3 = error_kappa_table_2_v2[['Imprecision', 'Bias','Agreement']].apply(lambda x: x*100) # percentage unit conversion
                error_kappa_table_2_v3 = pd.concat([error_kappa_table_2_v3, error_kappa_table_2_v2[['Agreement Category']]], axis = 1) # concat 
                ### Specificity percentage unit conversion by multiplying 100
                error_specificity_table_3_v3 = error_specificity_table_3[['Imprecision', 'Bias','Specificity']].apply(lambda x: x*100) # percentage unit conversion
                error_specificity_table_3_v3 = pd.concat([error_specificity_table_3_v3, error_specificity_table_3[['Specificity Category']]], axis = 1) # concat 
                ### Specificity percentage unit conversion by multiplying 100
                error_sensitivity_table_3_v3 = error_sensitivity_table_3[['Imprecision', 'Bias','Sensitivity']].apply(lambda x: x*100) # percentage unit conversion
                error_sensitivity_table_3_v3 = pd.concat([error_sensitivity_table_3_v3, error_sensitivity_table_3[['Sensitivity Category']]], axis = 1) # concat 
                
                
                # Merge the three dataframes on 'Imprecision' and 'Bias' columns
                merged_df = pd.merge(error_specificity_table_3_v3, error_sensitivity_table_3_v3[['Imprecision', 'Bias', 'Sensitivity', 'Sensitivity Category']], on=['Imprecision', 'Bias'])
                merged_df = pd.merge(merged_df, error_kappa_table_2_v3[['Imprecision', 'Bias', 'Agreement', 'Agreement Category']], on=['Imprecision', 'Bias'])             
             
                                
                                
                with tab3:
                    #merged_df
                    # Use existing categorical columns for plotting
                    agreement_categories = merged_df['Agreement Category'].unique()
                    sensitivity_categories = merged_df['Sensitivity Category'].unique()
                    specificity_categories = merged_df['Specificity Category'].unique()
                                       
                    # Create a 3D scatter plot with different colors for each metric
                    fig555 = go.Figure()

                    # Add Agreement points
                    fig555.add_trace(go.Scatter3d(
                        x=merged_df['Bias'],
                        y=merged_df['Imprecision'],
                        z=merged_df['Agreement'],
                        mode='markers',
                        marker=dict(size=5, color=merged_df['Agreement'], colorscale='Viridis', opacity=0.8),
                        name='Agreement',
                        hovertemplate='Bias: %{x}<br>Imprecision: %{y}<br>Agreement: %{z}<extra></extra>'
                    ))


                    # Add horizontal planes to distinguish Agreement categories (e.g., < threshold and ≥ threshold)
                    
                    fig555.add_trace(go.Surface(
                        z=[[min_agg_threshold, min_agg_threshold], [min_agg_threshold, min_agg_threshold]],
                        x=[merged_df['Bias'].min(), merged_df['Bias'].max()],
                        y=[merged_df['Imprecision'].min(), merged_df['Imprecision'].max()],
                        colorscale='Reds',
                        opacity=0.3,
                        showscale=False,
                        name=f'Agreement Threshold {min_agg_threshold}'
                    ))

                    
                    fig555.add_trace(go.Surface(
                        z=[[des_agg_threshold, des_agg_threshold], [des_agg_threshold, des_agg_threshold]],
                        x=[merged_df['Bias'].min(), merged_df['Bias'].max()],
                        y=[merged_df['Imprecision'].min(), merged_df['Imprecision'].max()],
                        colorscale='Greens',
                        opacity=0.3,
                        showscale=False,
                        name=f'Agreement Threshold {des_agg_threshold}'
                    ))

                    
                    fig555.add_trace(go.Surface(
                        z=[[opt_agg_threshold, opt_agg_threshold], [opt_agg_threshold, opt_agg_threshold]],
                        x=[merged_df['Bias'].min(), merged_df['Bias'].max()],
                        y=[merged_df['Imprecision'].min(), merged_df['Imprecision'].max()],
                        colorscale='Blues',
                        opacity=0.3,
                        showscale=False,
                        name=f'Agreement Threshold {opt_agg_threshold}'
                    ))

                    # Update layout for better visualization
                    fig555.update_layout(
                        title='3D Scatter Plot of Bias, Imprecision, and Metrics',
                        scene=dict(
                            xaxis_title='Bias (%)',
                            yaxis_title='Imprecision (%)',
                            zaxis_title='Agreement (%)',
                            #zaxis=dict(
                             #           range=[min_agg_threshold - 10, merged_df['Agreement'].max()]  # Set the z-axis range
                              #      ),
                        ),
                        legend_title='Metrics'
                    )
                    
                    st.write("##### **:blue[APS Plots for imprecision based on overall agreement]**")
                    # Show figure in Streamlit
                    st.plotly_chart(fig555, theme="streamlit", use_container_width=True)

                    
                    





                    # Assign a unique numeric value to each category for color mapping
                    #merged_df['Category_Num'] = merged_df['Agreement Category'].factorize()[0]

                    # Create a Plotly Express scatter plot with automatic color assignment based on 'Agreement Category'
                    fig = px.scatter(
                        merged_df,
                        x='Bias',
                        y='Imprecision',
                        color='Agreement Category',
                        color_discrete_sequence=px.colors.qualitative.T10,  # Choose a color scheme here
                        category_orders={"Agreement Category": merged_df['Agreement Category'].unique()},  # Ensure consistent category order
                        title='Hexbin-like Plot of Bias vs. Imprecision with Agreement Categories'
                    )

                    # Update marker size and style for better visual distinction
                    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='black')), mode='markers')

                    # Customize layout further if needed
                    fig.update_layout(
                        xaxis_title='Bias',
                        yaxis_title='Imprecision',
                        legend_title='Agreement Category',
                        template='plotly_white'
                    )

                    # Show the plot in Streamlit
                    st.plotly_chart(fig, theme="streamlit", use_container_width=True)











                    ### interactive contour plot
                    st.write("##### **:blue[APS for imprecision based on overall agreement]**")
                    ###### interactive plot

                    limit_xn = min_agg_threshold - 5
                    limit_xp = 100
                    ylim_v1 = merged_df[merged_df['Agreement'] >= min_agg_threshold - 5]['Imprecision'].max()

                    # min, desirable, optimal imprecision limits
                    ylim_v2 = round(merged_df[merged_df['Agreement'] >= min_agg_threshold]['Imprecision'].max(), 1)
                    ylim_v3 = round(merged_df[merged_df['Agreement'] >= des_agg_threshold]['Imprecision'].max(), 1)
                    ylim_v4 = round(merged_df[merged_df['Agreement'] >= opt_agg_threshold]['Imprecision'].max(), 1)


                    # min, desirable, optimal bias limits
                    ylim_v5 = round(merged_df[merged_df['Agreement'] >= min_agg_threshold]['Bias'].max(), 1)
                    ylim_v6 = round(merged_df[merged_df['Agreement'] >= des_agg_threshold]['Bias'].max(), 1)
                    ylim_v7 = round(merged_df[merged_df['Agreement'] >= opt_agg_threshold]['Bias'].max(), 1)
                    

                    # min, desirable, optimal bias limits
                    ylim_v8 = round(merged_df[merged_df['Agreement'] >= min_agg_threshold]['Bias'].min(), 1)
                    ylim_v9 = round(merged_df[merged_df['Agreement'] >= des_agg_threshold]['Bias'].min(), 1)
                    ylim_v10 = round(merged_df[merged_df['Agreement'] >= opt_agg_threshold]['Bias'].min(), 1)
                                                            
                    # Define the colors based on the thresholds
                    category_colors = {
                        f'≥{min_agg_threshold}%': 'rgb(166, 0, 0)',    # Dark Red
                        f'≥{des_agg_threshold}%': 'rgb(0, 176, 0)',    # Green
                        f'≥{opt_agg_threshold}%': 'rgb(0, 0, 255)',    # Blue
                    }

                    # Text info of APS
                    na_quote_1 = ' '
                    na_quote_2 = ' '
                    # >33% MU is unaccceptable
                    if ylim_v2 > 33:
                        ylim_v2 = "NA"
                        na_quote_1 = "NA: Not available"
                    if ylim_v3 > 33:
                        ylim_v3 = "NA"
                        na_quote_1 = "NA: Not available"
                    if ylim_v4 > 33:
                        ylim_v4 = "NA"
                        na_quote_1 = "NA: Not available"
                    if ylim_v5 > 33:
                        ylim_v5 = "NA"
                        na_quote_1 = "NA: Not available"
                    if ylim_v6 > 33:
                        ylim_v6 = "NA"
                        na_quote_1 = "NA: Not available"
                    if ylim_v7 > 33:
                        ylim_v7 = "NA"
                        na_quote_1 = "NA: Not available"  
                    if ylim_v8 > 33:
                        ylim_v8 = "NA"
                        na_quote_1 = "NA: Not available"
                    if ylim_v9 > 33:
                        ylim_v9 = "NA"
                        na_quote_1 = "NA: Not available"
                    if ylim_v10 > 33:
                        ylim_v10 = "NA"
                        na_quote_1 = "NA: Not available"                                           
                    
                    # =0% MU is not obtainable
                    if ylim_v2 == 0 or np.isnan(ylim_v2):
                        ylim_v2 = "NO"
                        na_quote_2 = "NO: Not obtainable"
                    if ylim_v3 == 0 or np.isnan(ylim_v3):
                        ylim_v3 = "NO"
                        na_quote_2 = "NO: Not obtainable"
                    if ylim_v4 == 0 or np.isnan(ylim_v4):
                        ylim_v4 = "NO"
                        na_quote_2 = "NO: Not obtainable"
                    if ylim_v5 == 0 or np.isnan(ylim_v5):
                        ylim_v5 = "NO"
                        na_quote_2 = "NO: Not obtainable"
                    if ylim_v6 == 0 or np.isnan(ylim_v6):
                        ylim_v6 = "NO"
                        na_quote_2 = "NO: Not obtainable"
                    if ylim_v7 == 0 or np.isnan(ylim_v7):
                        ylim_v7 = "NO"
                        na_quote_2 = "NO: Not obtainable"                        
                    if ylim_v8 == 0 or np.isnan(ylim_v8):
                        ylim_v8 = "NO"
                        na_quote_2 = "NO: Not obtainable"
                    if ylim_v9 == 0 or np.isnan(ylim_v9):
                        ylim_v9 = "NO"
                        na_quote_2 = "NO: Not obtainable"
                    if ylim_v10 == 0 or np.isnan(ylim_v10):
                        ylim_v10 = "NO"
                        na_quote_2 = "NO: Not obtainable"                               
                    st.write(" ")
                    st.markdown(f"""
                                | APS level | Imprecision | Positive Bias | Negative Bias |
                                | ----------- | ----------- | ----------- | ----------- |
                                | :red[Minimum] | {ylim_v2} | {ylim_v5} | {ylim_v8} |
                                | :green[Desirable] | {ylim_v3} | {ylim_v6} | {ylim_v9} |
                                | :blue[Optimal] | {ylim_v4} | {ylim_v7} | {ylim_v10} |
                                """)
                    st.write(na_quote_1) # >33% MU is unaccceptable
                    st.write(na_quote_2) # =0% MU is notobtainable
                    st.markdown("---")
                
                with tab4:
                    st.write('##### **:blue[APS for Imprecision and Bias based on sublevel aggreement]**')
                    st.write('  ')
                    
                    for i in names: 
                        st.write('  ')
                        st.markdown(f'##### **:blue[APS based on sublevel: {i}]**')
                        level_name = i
                        
                        ### Agreement over-sub
                        cat_Agree= pd.cut(error_kappa_table_2[level_name], bins_k, labels=names_k)
                        cat_Agree_df= pd.DataFrame(cat_Agree)
                        # concat dfs
                        cat_Agree_df = cat_Agree_df.rename(columns={level_name: "Agreement Category"})
                        # Outcome full
                        error_kappa_table_2_v2 = pd.concat([error_kappa_table_2, cat_Agree_df],axis = 1)                
                        # percentage unit conversion by multiplying 100
                        error_kappa_table_2_v3 = error_kappa_table_2_v2[['Imprecision', 'Bias',level_name]].apply(lambda x: x*100) # percentage unit conversion
                        error_kappa_table_2_v3 = pd.concat([error_kappa_table_2_v3, error_kappa_table_2_v2[['Agreement Category']]], axis = 1) # concat 
                                            
                        ### Sensitivity over-sub
                        cat_Sens= pd.cut(error_sensitivity_table_2[level_name], bins_k, labels=names_k)
                        cat_Sens_df= pd.DataFrame(cat_Sens)
                        # concat dfs
                        cat_Sens_df = cat_Sens_df.rename(columns={level_name: "Sensitivity Category"})
                        # Outcome full
                        error_sensitivity_table_2_v2 = pd.concat([error_sensitivity_table_2, cat_Sens_df],axis = 1)                
                        # percentage unit conversion by multiplying 100
                        error_sensitivity_table_2_v3 = error_sensitivity_table_2_v2[['Imprecision', 'Bias' ,level_name]].apply(lambda x: x*100) # percentage unit conversion
                        error_sensitivity_table_2_v3 = pd.concat([error_sensitivity_table_2_v3, error_sensitivity_table_2_v2[['Sensitivity Category']]], axis = 1) # concat 

                        ### Specificity over-sub
                        cat_Spec= pd.cut(error_specificity_table_2[level_name], bins_k, labels=names_k)
                        cat_Spec_df= pd.DataFrame(cat_Spec)
                        # concat dfs
                        cat_Spec_df = cat_Spec_df.rename(columns={level_name: "Specificity Category"})
                        # Outcome full
                        error_specificity_table_2_v2 = pd.concat([error_specificity_table_2, cat_Spec_df],axis = 1)                
                        # percentage unit conversion by multiplying 100
                        error_specificity_table_2_v3 = error_specificity_table_2_v2[['Imprecision', 'Bias',level_name]].apply(lambda x: x*100) # percentage unit conversion
                        error_specificity_table_2_v3 = pd.concat([error_specificity_table_2_v3, error_specificity_table_2_v2[['Specificity Category']]], axis = 1) # concat 

                        #error_kappa_table_2_v3
                        #error_sensitivity_table_2_v3
                        #error_specificity_table_2_v3
                        
                        # Merge the three dataframes on 'Imprecision' and 'Bias' columns
                        merged_df2 = pd.merge(error_specificity_table_2_v3, error_sensitivity_table_2_v3[['Imprecision', 'Bias', 'Sensitivity Category']], on=['Imprecision', 'Bias'])
                        merged_df2 = pd.merge(merged_df2, error_kappa_table_2_v3[['Imprecision', 'Bias', 'Agreement Category']], on=['Imprecision', 'Bias'])             
                        # level_name
                        
                        
                        # Create a 3D scatter plot with different colors for each metric
                        fig555 = go.Figure()

                        # Add Agreement points
                        fig555.add_trace(go.Scatter3d(
                            x=error_kappa_table_2_v3['Bias'],
                            y=error_kappa_table_2_v3['Imprecision'],
                            z=error_kappa_table_2_v3[level_name],
                            mode='markers',
                            marker=dict(size=5, color=error_kappa_table_2_v3[level_name], colorscale='Viridis', opacity=0.8),
                            name='Agreement',
                            hovertemplate='Bias: %{x}<br>Imprecision: %{y}<br>Agreement: %{z}<extra></extra>'
                        ))

                        # Add Sensitivity points
                        fig555.add_trace(go.Scatter3d(
                            x=error_sensitivity_table_2_v3['Bias'],
                            y=error_sensitivity_table_2_v3['Imprecision'],
                            z=error_sensitivity_table_2_v3[level_name],
                            mode='markers',
                            marker=dict(size=5, color=error_sensitivity_table_2_v3[level_name], colorscale='Agsunset', opacity=0.8),
                            name='Sensitivity',
                            hovertemplate='Bias: %{x}<br>Imprecision: %{y}<br>Sensitivity: %{z}<extra></extra>'
                        ))

                        # Add Specificity points
                        fig555.add_trace(go.Scatter3d(
                            x=error_specificity_table_2_v3['Bias'],
                            y=error_specificity_table_2_v3['Imprecision'],
                            z=error_specificity_table_2_v3[level_name],
                            mode='markers',
                            marker=dict(size=5, color=error_specificity_table_2_v3[level_name], colorscale='Reds', opacity=0.8),
                            name='Specificity',
                            hovertemplate='Bias: %{x}<br>Imprecision: %{y}<br>Specificity: %{z}<extra></extra>'
                        ))

                        # Add horizontal planes to distinguish Agreement categories (e.g., < threshold and ≥ threshold)
                        
                        fig555.add_trace(go.Surface(
                            z=[[min_agg_threshold, min_agg_threshold], [min_agg_threshold, min_agg_threshold]],
                            x=[merged_df2['Bias'].min(), merged_df2['Bias'].max()],
                            y=[merged_df2['Imprecision'].min(), merged_df2['Imprecision'].max()],
                            colorscale='Reds',
                            opacity=0.3,
                            showscale=False,
                            name=f'Agreement Threshold {min_agg_threshold}'
                        ))

                        
                        fig555.add_trace(go.Surface(
                            z=[[des_agg_threshold, des_agg_threshold], [des_agg_threshold, des_agg_threshold]],
                            x=[merged_df2['Bias'].min(), merged_df2['Bias'].max()],
                            y=[merged_df2['Imprecision'].min(), merged_df2['Imprecision'].max()],
                            colorscale='Greens',
                            opacity=0.3,
                            showscale=False,
                            name=f'Agreement Threshold {des_agg_threshold}'
                        ))

                        
                        fig555.add_trace(go.Surface(
                            z=[[opt_agg_threshold, opt_agg_threshold], [opt_agg_threshold, opt_agg_threshold]],
                            x=[merged_df2['Bias'].min(), merged_df2['Bias'].max()],
                            y=[merged_df2['Imprecision'].min(), merged_df2['Imprecision'].max()],
                            colorscale='Blues',
                            opacity=0.3,
                            showscale=False,
                            name=f'Agreement Threshold {opt_agg_threshold}'
                        ))

                        # Update layout for better visualization
                        fig555.update_layout(
                            title='3D Scatter Plot of Bias, Imprecision, and Metrics',
                            scene=dict(
                                xaxis_title='Bias',
                                yaxis_title='Imprecision',
                                zaxis_title='Metric Value'
                            ),
                            legend_title='Metrics'
                        )

                        # Show figure in Streamlit
                        st.plotly_chart(fig555, theme="streamlit", use_container_width=True)
                        





                        fig123 = px.scatter(
                            merged_df2,
                            x='Bias',
                            y='Imprecision',
                            color='Agreement Category',
                            color_discrete_sequence=px.colors.qualitative.T10,  # Choose a color scheme here
                            category_orders={"Agreement Category": merged_df2['Agreement Category'].unique()},  # Ensure consistent category order
                            title='Hexbin-like Plot of Bias vs. Imprecision with Agreement Categories'
                        )

                        # Update marker size and style for better visual distinction
                        fig123.update_traces(marker=dict(size=10, line=dict(width=1, color='black')), mode='markers')

                        # Customize layout further if needed
                        fig123.update_layout(
                            xaxis_title='Bias',
                            yaxis_title='Imprecision',
                            legend_title='Agreement Category',
                            template='plotly_white'
                        )

                        # Show the plot in Streamlit
                        st.plotly_chart(fig123, theme="streamlit", use_container_width=True)










                        # CAUTION
                        ### interactive contour plot
                        ###### interactive plot

                        limit_xn = min_agg_threshold - 5
                        limit_xp = 100
                        ylim_v1 = merged_df[merged_df['Agreement'] >= min_agg_threshold - 5]['Imprecision'].max()

                        # min, desirable, optimal imprecision limits
                        ylim_v2 = round(error_kappa_table_2_v3[error_kappa_table_2_v3[level_name] >= min_agg_threshold]['Imprecision'].max(), 1)
                        ylim_v3 = round(error_kappa_table_2_v3[error_kappa_table_2_v3[level_name] >= des_agg_threshold]['Imprecision'].max(), 1)
                        ylim_v4 = round(error_kappa_table_2_v3[error_kappa_table_2_v3[level_name] >= opt_agg_threshold]['Imprecision'].max(), 1)


                        # min, desirable, optimal bias limits
                        ylim_v5 = round(error_kappa_table_2_v3[error_kappa_table_2_v3[level_name] >= min_agg_threshold]['Bias'].max(), 1)
                        ylim_v6 = round(error_kappa_table_2_v3[error_kappa_table_2_v3[level_name] >= des_agg_threshold]['Bias'].max(), 1)
                        ylim_v7 = round(error_kappa_table_2_v3[error_kappa_table_2_v3[level_name] >= opt_agg_threshold]['Bias'].max(), 1)
                        

                        # min, desirable, optimal bias limits
                        ylim_v8 = round(error_kappa_table_2_v3[error_kappa_table_2_v3[level_name] >= min_agg_threshold]['Bias'].min(), 1)
                        ylim_v9 = round(error_kappa_table_2_v3[error_kappa_table_2_v3[level_name] >= des_agg_threshold]['Bias'].min(), 1)
                        ylim_v10 = round(error_kappa_table_2_v3[error_kappa_table_2_v3[level_name] >= opt_agg_threshold]['Bias'].min(), 1)
                                                                

                        # Text info of APS
                        na_quote_1 = ' '
                        na_quote_2 = ' '
                        # >33% MU is unaccceptable
                        if ylim_v2 > 33:
                            ylim_v2 = "NA"
                            na_quote_1 = "NA: Not available"
                        if ylim_v3 > 33:
                            ylim_v3 = "NA"
                            na_quote_1 = "NA: Not available"
                        if ylim_v4 > 33:
                            ylim_v4 = "NA"
                            na_quote_1 = "NA: Not available"
                        if ylim_v5 > 33:
                            ylim_v5 = "NA"
                            na_quote_1 = "NA: Not available"
                        if ylim_v6 > 33:
                            ylim_v6 = "NA"
                            na_quote_1 = "NA: Not available"
                        if ylim_v7 > 33:
                            ylim_v7 = "NA"
                            na_quote_1 = "NA: Not available"  
                        if ylim_v8 > 33:
                            ylim_v8 = "NA"
                            na_quote_1 = "NA: Not available"
                        if ylim_v9 > 33:
                            ylim_v9 = "NA"
                            na_quote_1 = "NA: Not available"
                        if ylim_v10 > 33:
                            ylim_v10 = "NA"
                            na_quote_1 = "NA: Not available"                                           
                        
                        # =0% MU is not obtainable
                        if ylim_v2 == 0 or np.isnan(ylim_v2):
                            ylim_v2 = "NO"
                            na_quote_2 = "NO: Not obtainable"
                        if ylim_v3 == 0 or np.isnan(ylim_v3):
                            ylim_v3 = "NO"
                            na_quote_2 = "NO: Not obtainable"
                        if ylim_v4 == 0 or np.isnan(ylim_v4):
                            ylim_v4 = "NO"
                            na_quote_2 = "NO: Not obtainable"
                        if ylim_v5 == 0 or np.isnan(ylim_v5):
                            ylim_v5 = "NO"
                            na_quote_2 = "NO: Not obtainable"
                        if ylim_v6 == 0 or np.isnan(ylim_v6):
                            ylim_v6 = "NO"
                            na_quote_2 = "NO: Not obtainable"
                        if ylim_v7 == 0 or np.isnan(ylim_v7):
                            ylim_v7 = "NO"
                            na_quote_2 = "NO: Not obtainable"                        
                        if ylim_v8 == 0 or np.isnan(ylim_v8):
                            ylim_v8 = "NO"
                            na_quote_2 = "NO: Not obtainable"
                        if ylim_v9 == 0 or np.isnan(ylim_v9):
                            ylim_v9 = "NO"
                            na_quote_2 = "NO: Not obtainable"
                        if ylim_v10 == 0 or np.isnan(ylim_v10):
                            ylim_v10 = "NO"
                            na_quote_2 = "NO: Not obtainable"                               
                        st.write(" ")
                        st.markdown(f"""
                                    | APS level | Imprecision | Positive Bias | Negative Bias |
                                    | ----------- | ----------- | ----------- | ----------- |
                                    | :red[Minimum] | {ylim_v2} | {ylim_v5} | {ylim_v8} |
                                    | :green[Desirable] | {ylim_v3} | {ylim_v6} | {ylim_v9} |
                                    | :blue[Optimal] | {ylim_v4} | {ylim_v7} | {ylim_v10} |
                                    """)
                        st.write(na_quote_1) # >33% MU is unaccceptable
                        st.write(na_quote_2) # =0% MU is notobtainable
                        st.markdown("---")               
                
                placeholder.success('**Done**', icon="✅")
                time.sleep(2)
                placeholder.empty()




            
    except IndexError as error:
        #print("NameError occurred:", error)
        #st.write(error)
        st.error('The number of measurand results belonging to a particular category may be too low for determination of APS for MU. Please check your clinical decision limits or upload a larger data set.', icon="❗")   
    except NameError as error:
        #print("NameError occurred:", error)
        #st.write(error)
        st.error('Please upload your file')
    except ValueError as error:
        #print("ValueError occurred:", error)
        #st.write(error)
        st.error('Inappropriate clinical decision limit was entered.', icon="❗")
    except TypeError as error:
        #st.write(error)
        st.error('Your data includes non-numerical types of entry. Please check your data.', icon="❗")
    except Exception as error:
        #print("An exception occurred:", error)
        #st.write(error)
        st.info('Upload your file and follow the instructions to calculate APS', icon = "📁")
        
