# Udacity Machine Learning Engineer with Microsoft Azure Capstone Project

## Table of Contents
* [Overview](#Overview)
* [Dataset](#Dataset)
    * [Dataset Overview](#Dataset-Overview)
    * [Data dictionary](#Data-dictionary)
    * [Sector catalogue](#Sector-catalogue)
    * [Entities catalogue](#Entities-catalogue)
    * [Yes-No catalogue](#Yes-No-catalogue)
    * [Laboratory result catalogue](#Laboratory-result-catalogue)
    * [Final Classification catalogue](#Final-Classification-catalogue)
## Overview

This is the capstone project for the Udacity Machine Learning Engineer with Microsoft Azure.

In this project I analize the Mexican Government's data for the COVID-19 pandemic from January of 2020 up until the 30th of April of 2021. From this data I intend to get a predictive model to analyze a person's probability to enter an Intensive Care Unit (ICU) based on their COVID-19 test result type, age, gender and comorbidities.

This is the project workflow that was followed.
![Capstone project diagram](images/capstone-diagram.png)

1. Choose a dataset: The dataset chosen for the project is the Mexican Government's General Directorate of Epidemiology COVID-19 Open Data. Available to download in the following URL: https://www.gob.mx/salud/documentos/datos-abiertos-152127
2. Import Dataset into workspace: The dataset in CSV format is registered in the Datasets tab in Azure ML Studio to be used for training.
3. Train model using Automated ML: Using the AzureML SDK for Python a Jupyter Notebook is created where a classification model using AutoML is trained, and the best one is selected.
4. Train model using HyperDrive: Using the AzureML SDK for Python a Jupyter Notebook is created where a classification model using HyperDrive for hyperparameter optimization is trained.
5. Compare model performance: The model with the best accuracy is selected for deployment.
6. Deploy best model: The best model is deployed using Azure Container Instances, a functional endpoint is produced and logging is enabled with Application Insights.
7. Test model endpoint: The endpoint is tested with Apache Benchmark.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Dataset Overview
As explained before, the dataset is from the Mexican Government's General Directorate of Epidemiology COVID-19 Open Data. We are given a description for each column of the dataset in a dictionary, and the posible values for each column in catalogues.

### Data dictionary
| # | Variable Name         | Variable Description (With English translation) | Format |
|---|-----------------------|----------------------|--------|
|1  |FECHA_ACTUALIZACION    |(Update date) The last date the database was updated, this variable allows to identify the date of the last update.                     |AAAA-MM-DD        |
|2  |ID_REGISTRO            |(Registry ID) Case identifier number                      |Text        |
|3  |ORIGEN                 |(Origin) Sentinel surveillance is carried out through the respiratory disease monitoring health unit system (USMER). The USMER include medical units of the first, second or third level of care, and third level units also participate as USMERs, which due to their characteristics contribute to broadening the epidemiological information panorama, including those with a specialty in pulmonology, infectology or pediatrics. . (Categories in Annex Catalog).                      | 1 = USMER, 2 = Outside USMER, 99 = Not Specified        |
|4  |SECTOR                 |(Sector) Identifies the type of institution of the National Health System that provided the care.                      |[Sector catalogue](#Sector-catalogue)        |
|5  |ENTIDAD_UM             |(Entity) Identifies the entity (state) where the medical unit that provided the care is located.                      |[Entities catalogue](#Entities-catalogue)        |
|6  |SEXO                   |(Sex) Identifies the sex of the patient.                      |1 = Woman, 2 = Man, 99 = Not specified     |
|7  |ENTIDAD_NAC            |(Entity of birth) Identifies the patient's birth entity (state).                     |[Entities catalogue](#Entities-catalogue)        |
|8  |ENTIDAD_RES            |(Entity of residence) Identifies the entity (state) of residence of the patient.                      |[Entities catalogue](#Entities-catalogue)        |
|9  |MUNICIPIO_RES          |(Municipality of residence) Identifies the municipality of residence of the patient.                     | *Catalogue has 2,500 rows and was not included for simplicity, but 997 = Does not apply, 998 = Ignored, 999 = Not specified       |
|10 |TIPO_PACIENTE          |(Patient care type) Identifies the type of care the patient received in the unit. It is called ambulatory if the patient returned home or it is called hospitalized if it was admitted to the hospital.                      | 1 = Ambulatory, 2 = Hospitalized, 99 = Not specified    |
|11 |FECHA_INGRESO          |(Date of entry) Identifies the date of admission of the patient to the care unit.                      |AAAA-MM-DD        |
|12 |FECHA_SINTOMAS         |(Date of symptoms) Identifies the date on which the patient's symptoms began.                    |AAAA-MM-DD        |
|13 |FECHA_DEF              |(Date of death) Identifies the date the patient died if it did. If it did not die the date is displayed as 9999-99-99                      |AAAA-MM-DD        |
|14 |INTUBADO               |(Intubated) Identifies if the patient required intubation.                      |[Yes-No catalogue](#Yes-No-catalogue)        |
|15 |NEUMONIA               |(Pneumonia) Identifies if the patient was diagnosed with pneumonia.                     |[Yes-No catalogue](#Yes-No-catalogue)        |
|16 |EDAD                   |(Age) Identifies the age of the patient                      | Numeric in years       |
|17 |NACIONALIDAD           |(Nationality) Identifies if the patient is Mexican or foreign.                      | 1 = Mexican, 2 = Foreign, 99 = Not specified       |
|18 |EMBARAZO               |(Pregnancy) Identifies if the patient is pregnant.                     |[Yes-No catalogue](#Yes-No-catalogue)        |
|19 |HABLA_LENGUA_INDIG     |(Speaks indigenous language) Identifies if the patient speaks an indigenous language.                      |[Yes-No catalogue](#Yes-No-catalogue)        |
|20 |INDIGENA               |(Indigenous person) Identifies if the patient self-identifies as an indigenous person.                      |[Yes-No catalogue](#Yes-No-catalogue)        |
|21 |DIABETES               |(Diabetes) Identifies if the patient has a diagnosis of diabetes.                      |[Yes-No catalogue](#Yes-No-catalogue)        |
|22 |EPOC                   |(COPD) Identifies if the patient has a COPD (Chronic Obstructive Pulmonary Disease) diagnosis.                      |[Yes-No catalogue](#Yes-No-catalogue)        |
|23 |ASMA                   |(Asthma) Identifies if the patient has a diagnosis of asthma.                     |[Yes-No catalogue](#Yes-No-catalogue)        |
|24 |INMUSUPR               |(Immunosuppression) Identifies if the patient is immunosuppressed.                      |[Yes-No catalogue](#Yes-No-catalogue)        |
|25 |HIPERTENSION           |(Hypertension) Identifies if the patient has a diagnosis of hypertension.                     |[Yes-No catalogue](#Yes-No-catalogue)        |
|26 |OTRAS_COM              |(Other comorbidities) Identifies if the patient has a diagnosis of other diseases.                     |[Yes-No catalogue](#Yes-No-catalogue)        |
|27 |CARDIOVASCULAR         |(Cardiovascular disease) Identifies if the patient has a diagnosis of cardiovascular disease.                     |[Yes-No catalogue](#Yes-No-catalogue)        |
|28 |OBESIDAD               |(Obesity) Identifies if the patient has a diagnosis of obesity.                      | [Yes-No catalogue](#Yes-No-catalogue)       |
|29 |RENAL_CRONICA          |(Chronic kidney failure) Identifies if the patient has a diagnosis of chronic kidney failure.                      |[Yes-No catalogue](#Yes-No-catalogue)        |
|30 |TABAQUISMO             |(Smoking) Identifies if the patient has a smoking habit.                      |[Yes-No catalogue](#Yes-No-catalogue)        |
|31 |OTRO_CASO              |(Other case) Identifies if the patient had contact with any other case diagnosed with SARS CoV-2                      |[Yes-No catalogue](#Yes-No-catalogue)        |
|32 |TOMA_MUESTRA_LAB       |(Laboratory sample taken) Identifies if the patient had a laboratory sample taken.                      | [Yes-No catalogue](#Yes-No-catalogue)       |
|33 |RESULTADO_LAB          |(Laboratory sample result) Identifies the result of the analysis of the sample reported by the laboratory of the National Network of Epidemiological Surveillance Laboratories (INDRE, LESP and LAVE) and private laboratories endorsed by InDRE whose results are registered in SISVER.                      |[Laboratory result catalogue](#Laboratory-result-catalogue)        |
|34 |TOMA_MUESTRA_ANTIGENO  |(Antigen sample taken) Identifies if the patient had an antigen sample for SARS-CoV-2                      |[Yes-No catalogue](#Yes-No-catalogue)        |
|35 |RESULTADO_ANTIGENO     |(Antigen sample result) Identifies the result of the analysis of the antigen sample taken from the patient                      | 1 = Positive to SARS-CoV-2, 2 = Negative to SARS-CoV-2, 99 = Does not apply (case without sample)      |
|36 |CLASIFICACION_FINAL    |(Final classification) Identifies if the patient is a case of COVID-19 according to the Final classification catalog.                      |[Final Classification catalogue](#Final-Classification-catalogue)        |
|37 |MIGRANTE               |(Migrant) Identifies if the patient is a migrant person.                      |[Yes-No catalogue](#Yes-No-catalogue)        |
|38 |PAIS_NACIONALIDAD      |(Nationality) Identifies the nationality of the patient.                      |Text, 99 = Ignore        |
|39 |PAIS_ORIGEN            |(Country of origin) Identifies the country from which the patient departed for Mexico.                      |Text, 97 = Does not apply        |
|40 |UCI                    |(ICU) Identifies if the patient required admission to an Intensive Care Unit.                      |[Yes-No catalogue](#Yes-No-catalogue)        |

### Sector catalogue
| Key | Value |
|-----|-------|
|1    |CRUZ ROJA       |
|2    |DIF       |
|3    |ESTATAL       |
|4    |IMSS       |
|5    |IMSS-BIENESTAR       |
|6    |ISSSTE       |
|7    |MUNICIPAL       |
|8    |PEMEX       |
|9    |PRIVADA       |
|10   |SEDENA       |
|11   |SEMAR       |
|12   |SSA       |
|13   |UNIVERSITARIO       |
|99   |NO ESPECIFICADO       |

### Entities catalogue
| Key | Value | Abbreviation
|-----|-------|-----|
|01	|AGUASCALIENTES	|AS|
|02	|BAJA CALIFORNIA|	BC|
|03	|BAJA CALIFORNIA SUR|	BS|
|04	|CAMPECHE	|CC|
|05	|COAHUILA DE ZARAGOZA	|CL|
|06	|COLIMA	|CM|
|07	|CHIAPAS	|CS|
|08	|CHIHUAHUA	|CH|
|09	|CIUDAD DE MÉXICO	|DF|
|10	|DURANGO	|DG|
|11	|GUANAJUATO	|GT|
|12	|GUERRERO	|GR|
|13	|HIDALGO	|HG|
|14	|JALISCO	|JC|
|15	|MÉXICO	|MC|
|16	|MICHOACÁN 	|MN|
|17	|MORELOS	|MS|
|18	|NAYARIT	|NT|
|19	|NUEVO LEÓN	|NL|
|20	|OAXACA	|OC|
|21	|PUEBLA	|PL|
|22	|QUERÉTARO	|QT|
|23	|QUINTANA ROO	|QR|
|24	|SAN LUIS POTOSÍ	|SP|
|25	|SINALOA	|SL|
|26	|SONORA	|SR|
|27	|TABASCO	|TC|
|28	|TAMAULIPAS	|TS|
|29	|TLAXCALA	|TL|
|30	|VERACRUZ 	|VZ|
|31	|YUCATÁN	|YN|
|32	|ZACATECAS	|ZS|
|36	|ESTADOS UNIDOS MEXICANOS (UNITED MEXICAN STATES)	|EUM|
|97	|NO APLICA	(Does not apply)|NA|
|98	|SE IGNORA	(Ignored)|SI|
|99	|NO ESPECIFICADO	(Not specified)|NE|

### Yes-No catalogue
| Key | Value |
|-----|-------|
|1    |Yes       |
|2    |No       |
|97   |Des not apply       |
|98   |Ignored       |
|99   |Not specified       |

### Laboratory result catalogue
| Key | Value |
|-----|-------|
|1    |Positive to SARS-COV-2       |
|2    |Negative to SARS-COV-2       |
|3    |Pending result       |
|4    |Unsuitable result       |
|97   |Does not apply (case without sample)       |

## Final Classification catalogue
| Key | Value | Description
|-----|-------|-----|
|1    | COVID-19 CASE CONFIRMED BY EPIDEMIOLOGICAL CLINICAL ASSOCIATION | "Confirmed by association applies when the case reported being a positive contact for COVID-19 (and this is registered in SISVER) and: The case was not sampled or the sample was invalid. " |
|2    | COVID-19 CASE CONFIRMED BY DICTAMINATION COMMITTEE | "Confirmed by ruling only applies to deaths under the following conditions: The case was not sampled or a sample was taken, but the sample was invalid." |
|3    | CONFIRMED SARS-COV-2 CASE | "Confirmed applies when: The case has a laboratory sample or antigenic test and was positive for SARS-CoV-2, regardless of whether the case has a clinical epidemiological association. "|
|4    | INVALID BY LABORATORY | Invalid applies when the case does not have a clinical epidemiological association, nor a COVID-19 ruling. A laboratory sample was taken and it was invalid. |
|5    | NOT PERFORMED BY LABORATORY | Not carried out applies when the case does not have a clinical epidemiological association, nor a ruling on COVID-19 and a laboratory sample was taken and it was not processed. |
|6    | SUSPECT CASE | "Suspect applies when: The case does not have a clinical-epidemiological association, or a COVID-19 ruling and no sample was taken, or a laboratory sample was taken and the result is pending, regardless of another condition. " |
|7    | NEGATIVE TO SARS-COV-2 | "Negative applies when the case: 1. A laboratory sample was taken and it was: negative for SARS-COV-2 or positive for any other respiratory virus (Influenza, RSV, Bocavirus, others) regardless of whether this case has a clinical-epidemiological association or opinion to COVID-19. 2. An antigenic sample was taken that was negative for SARS-COV-2 and the case was not taken from a laboratory sample or confirmed by epidemiological association or by clinical epidemiological opinion. " |

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
The CSV file with all the data is uploaded to the Datasets tab in Azure ML Studio, there it can be accessed to the required Jupyter Notebooks.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

## References
- [Mexican Government's General Directorate of Epidemiology COVID-19 Open Data](https://www.gob.mx/salud/documentos/datos-abiertos-152127)
- [Udacity project starter files](https://github.com/udacity/nd00333-capstone/tree/master/starter_file)
