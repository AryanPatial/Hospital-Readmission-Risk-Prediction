-- One row per hospital stay -> model features + label.
--
-- Runs unchanged on DuckDB (local) and Athena (AWS). Both read a table called
-- raw_encounters where every column is text, exactly as the CSV ships, with column
-- names lowercased and '-' turned into '_' (Athena allows no hyphens; see
-- src/prepare.py raw_column_names). So only portable SQL here: CASE, CAST,
-- TRY_CAST, LIKE, SUBSTR, STRPOS. No engine-specific functions.
--
-- The same rules are written in Python in src/features.py for the live API.
-- tests/test_feature_parity.py checks both give identical output on every row.

SELECT
    CAST(encounter_id AS BIGINT)                                   AS encounter_id,
    CAST(patient_nbr  AS BIGINT)                                   AS patient_nbr,

    -- label: back in hospital within 30 days
    CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END                 AS readmit_30d,

    -- kept for the fairness audit only, NOT a model feature (see README)
    CASE WHEN race = '?' THEN 'Unknown' ELSE race END              AS race,

    gender,
    -- "[70-80)" -> 75 and "[0-10)" -> 5, the middle of the band
    CAST(SUBSTR(age, 2, STRPOS(age, '-') - 2) AS INTEGER) + 5      AS age_years,

    CASE WHEN admission_type_id IN ('1','2','3','7') THEN admission_type_id
         ELSE 'other' END                                          AS admission_type,
    CASE WHEN discharge_disposition_id = '1'  THEN 'home'
         WHEN discharge_disposition_id = '6'  THEN 'home_health'
         WHEN discharge_disposition_id IN ('3','4','24') THEN 'nursing_facility'
         WHEN discharge_disposition_id IN ('2','5','22','23') THEN 'other_inpatient'
         WHEN discharge_disposition_id = '7'  THEN 'left_ama'
         ELSE 'other' END                                          AS discharge_to,
    CASE WHEN admission_source_id = '7' THEN 'emergency'
         WHEN admission_source_id IN ('1','2','3') THEN 'referral'
         WHEN admission_source_id IN ('4','5','6','10','22','25') THEN 'transfer'
         ELSE 'other' END                                          AS admission_source,

    CAST(time_in_hospital   AS INTEGER)                            AS time_in_hospital,
    CAST(num_lab_procedures AS INTEGER)                            AS num_lab_procedures,
    CAST(num_procedures     AS INTEGER)                            AS num_procedures,
    CAST(num_medications    AS INTEGER)                            AS num_medications,
    CAST(number_outpatient  AS INTEGER)                            AS prior_outpatient,
    CAST(number_emergency   AS INTEGER)                            AS prior_emergency,
    CAST(number_inpatient   AS INTEGER)                            AS prior_inpatient,
    CAST(number_diagnoses   AS INTEGER)                            AS number_diagnoses,

    CASE WHEN medical_specialty = '?' THEN 'Unknown'
         WHEN medical_specialty IN ('InternalMedicine','Emergency/Trauma','Family/GeneralPractice',
              'Cardiology','Surgery-General','Nephrology','Orthopedics',
              'Orthopedics-Reconstructive','Radiologist','Pulmonology','Psychiatry')
              THEN medical_specialty
         ELSE 'Other' END                                          AS specialty,

    -- primary diagnosis -> ICD-9 chapter (Strack et al., 2014 grouping)
    CASE WHEN diag_1 LIKE 'V%' OR diag_1 LIKE 'E%' OR diag_1 = '?' THEN 'other'
         WHEN diag_1 LIKE '250%'                                          THEN 'diabetes'
         WHEN TRY_CAST(diag_1 AS DOUBLE) BETWEEN 390 AND 459.99
              OR TRY_CAST(diag_1 AS DOUBLE) BETWEEN 785 AND 785.99        THEN 'circulatory'
         WHEN TRY_CAST(diag_1 AS DOUBLE) BETWEEN 460 AND 519.99
              OR TRY_CAST(diag_1 AS DOUBLE) BETWEEN 786 AND 786.99        THEN 'respiratory'
         WHEN TRY_CAST(diag_1 AS DOUBLE) BETWEEN 520 AND 579.99
              OR TRY_CAST(diag_1 AS DOUBLE) BETWEEN 787 AND 787.99        THEN 'digestive'
         WHEN TRY_CAST(diag_1 AS DOUBLE) BETWEEN 800 AND 999.99           THEN 'injury'
         WHEN TRY_CAST(diag_1 AS DOUBLE) BETWEEN 710 AND 739.99           THEN 'musculoskeletal'
         WHEN TRY_CAST(diag_1 AS DOUBLE) BETWEEN 580 AND 629.99
              OR TRY_CAST(diag_1 AS DOUBLE) BETWEEN 788 AND 788.99        THEN 'genitourinary'
         WHEN TRY_CAST(diag_1 AS DOUBLE) BETWEEN 140 AND 239.99           THEN 'neoplasms'
         ELSE 'other' END                                          AS primary_diagnosis,

    CASE WHEN A1Cresult = 'None' THEN 'not_tested'
         WHEN A1Cresult = 'Norm' THEN 'normal'
         ELSE 'high' END                                           AS a1c_result,
    CASE WHEN max_glu_serum = 'None' THEN 'not_tested'
         WHEN max_glu_serum = 'Norm' THEN 'normal'
         ELSE 'high' END                                           AS glucose_result,
    insulin                                                        AS insulin,  -- No / Steady / Up / Down
    CASE WHEN change = 'Ch' THEN 1 ELSE 0 END                      AS meds_changed,
    CASE WHEN diabetesMed = 'Yes' THEN 1 ELSE 0 END                AS on_diabetes_meds,

    -- how many of the 23 diabetes drugs had their dose moved up or down
      CASE WHEN metformin   IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN repaglinide IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN nateglinide IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN chlorpropamide IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN glimepiride IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN acetohexamide IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN glipizide   IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN glyburide   IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN tolbutamide IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN pioglitazone IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN rosiglitazone IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN acarbose    IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN miglitol    IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN troglitazone IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN tolazamide  IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN examide     IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN citoglipton IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN insulin     IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN glyburide_metformin      IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN glipizide_metformin      IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN glimepiride_pioglitazone IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN metformin_rosiglitazone  IN ('Up','Down') THEN 1 ELSE 0 END
    + CASE WHEN metformin_pioglitazone   IN ('Up','Down') THEN 1 ELSE 0 END
                                                                   AS num_dose_changes

FROM raw_encounters
-- patients who died or went to hospice cannot be readmitted; 3 rows have no valid gender
WHERE discharge_disposition_id NOT IN ('11','13','14','19','20','21')
  AND gender IN ('Male','Female')
