from typing import Dict, List

from pydantic import BaseModel


class InferenceResult(BaseModel):
    SK_ID_CURR: int
    NAME: str
    pred_probs: float
    preds: int
    gt: int
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "SK_ID_CURR": 342407,
                    "NAME": "강민호",
                    "pred_probs": 0.16558438459907127,
                    "preds": 0,
                    "gt": 0,
                },
            ]
        }
    }


class InferenceDto(BaseModel):
    data: List[Dict] | None = None
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": [
                        {
                            "SK_ID_CURR": 342407,
                            "NAME": "강민호",
                            "TARGET": 0,
                            "NAME_CONTRACT_TYPE": "Cash loans",
                            "CODE_GENDER": "M",
                            "FLAG_OWN_CAR": "Y",
                            "FLAG_OWN_REALTY": "Y",
                            "CNT_CHILDREN": 0,
                            "AMT_INCOME_TOTAL": 360000.0,
                            "AMT_CREDIT": 1223010.0,
                            "AMT_ANNUITY": 48631.5,
                            "AMT_GOODS_PRICE": 1125000.0,
                            "NAME_TYPE_SUITE": "Unaccompanied",
                            "NAME_INCOME_TYPE": "State servant",
                            "NAME_EDUCATION_TYPE": "Secondary / secondary special",
                            "NAME_FAMILY_STATUS": "Married",
                            "NAME_HOUSING_TYPE": "House / apartment",
                            "REGION_POPULATION_RELATIVE": 0.022625,
                            "DAYS_BIRTH": -21486,
                            "DAYS_EMPLOYED": -13575,
                            "DAYS_REGISTRATION": -3582.0,
                            "DAYS_ID_PUBLISH": -4501,
                            "FLAG_MOBIL": 1,
                            "FLAG_EMP_PHONE": 1,
                            "FLAG_WORK_PHONE": 0,
                            "FLAG_CONT_MOBILE": 1,
                            "FLAG_PHONE": 0,
                            "FLAG_EMAIL": 0,
                            "CNT_FAM_MEMBERS": 2.0,
                            "REGION_RATING_CLIENT": 2,
                            "REGION_RATING_CLIENT_W_CITY": 2,
                            "WEEKDAY_APPR_PROCESS_START": "WEDNESDAY",
                            "HOUR_APPR_PROCESS_START": 18,
                            "REG_REGION_NOT_LIVE_REGION": 0,
                            "REG_REGION_NOT_WORK_REGION": 0,
                            "LIVE_REGION_NOT_WORK_REGION": 0,
                            "REG_CITY_NOT_LIVE_CITY": 0,
                            "REG_CITY_NOT_WORK_CITY": 1,
                            "LIVE_CITY_NOT_WORK_CITY": 1,
                            "EXT_SOURCE_2": 0.5435010500470568,
                            "EXT_SOURCE_3": 0.7517237147741489,
                            "OBS_30_CNT_SOCIAL_CIRCLE": 2.0,
                            "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
                            "OBS_60_CNT_SOCIAL_CIRCLE": 2.0,
                            "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
                            "DAYS_LAST_PHONE_CHANGE": -1986.0,
                            "FLAG_DOCUMENT_2": 0,
                            "FLAG_DOCUMENT_3": 0,
                            "FLAG_DOCUMENT_4": 0,
                            "FLAG_DOCUMENT_5": 0,
                            "FLAG_DOCUMENT_6": 0,
                            "FLAG_DOCUMENT_7": 0,
                            "FLAG_DOCUMENT_8": 1,
                            "FLAG_DOCUMENT_9": 0,
                            "FLAG_DOCUMENT_10": 0,
                            "FLAG_DOCUMENT_11": 0,
                            "FLAG_DOCUMENT_12": 0,
                            "FLAG_DOCUMENT_13": 0,
                            "FLAG_DOCUMENT_14": 0,
                            "FLAG_DOCUMENT_15": 0,
                            "FLAG_DOCUMENT_16": 0,
                            "FLAG_DOCUMENT_17": 0,
                            "FLAG_DOCUMENT_18": 0,
                            "FLAG_DOCUMENT_19": 0,
                            "FLAG_DOCUMENT_20": 0,
                            "FLAG_DOCUMENT_21": 0,
                            "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
                            "AMT_REQ_CREDIT_BUREAU_DAY": 0.0,
                            "AMT_REQ_CREDIT_BUREAU_WEEK": 0.0,
                            "AMT_REQ_CREDIT_BUREAU_MON": 0.0,
                            "AMT_REQ_CREDIT_BUREAU_QRT": 0.0,
                            "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
                            "AMT_ANNUITY_MAX": 17196.12,
                            "AMT_ANNUITY_SUM": 43966.259999999995,
                            "AMT_ANNUITY_MEAN": 14655.419999999998,
                            "AMT_CREDIT_MAX": 7920000.0,
                            "AMT_CREDIT_SUM": 18376614.0,
                            "AMT_CREDIT_MEAN": 1531384.5,
                            "AMT_CREDIT_MEDIAN": 897459.75,
                            "LOAN_COUNT": 12.0,
                            "OVERDUE_RATIO": 0.0,
                            "TOTAL_OVERDUE_COUNT": 0.0,
                            "AVERAGE_OVERDUE": 0.0,
                            "HAS_OVERDUE": 0.0,
                            "AMT_CREDIT_MAX_OVERDUE_MAX": 0.0,
                            "AMT_CREDIT_MAX_OVERDUE_SUM": 0.0,
                            "CNT_CREDIT_PROLONG_MAX": 0.0,
                            "CNT_CREDIT_PROLONG_SUM": 0.0,
                            "CNT_CREDIT_PROLONG_MEAN": 0.0,
                            "AMT_CREDIT_SUM_DEBT_MAX": 6390000.0,
                            "AMT_CREDIT_SUM_DEBT_SUM": 6390000.0,
                            "AMT_CREDIT_SUM_DEBT_MEAN": 798750.0,
                            "AMT_CREDIT_SUM_LIMIT_MEAN": 0.0,
                            "AMT_CREDIT_SUM_OVERDUE_MAX": 0.0,
                            "AMT_CREDIT_SUM_OVERDUE_SUM": 0.0,
                            "AMT_CREDIT_SUM_OVERDUE_MEAN": 0.0,
                            "COUNT_Overdue_0": 9.0,
                            "COUNT_Overdue_1": 0.0,
                            "COUNT_Overdue_2": 0.0,
                            "COUNT_Overdue_3": 0.0,
                            "COUNT_Overdue_4": 0.0,
                            "COUNT_Overdue_5": 0.0,
                            "NFLAG_INSURED_ON_APPROVAL_COUNT": 3.0,
                            "NFLAG_INSURED_ON_APPROVAL_SUM": 2.0,
                            "NFLAG_INSURED_ON_APPROVAL_RATIO": 0.6666666666666666,
                            "AMT_APPLICATION_MAX": 225000.0,
                            "AMT_APPLICATION_SUM": 502866.0,
                            "AMT_APPLICATION_MEAN": 167622.0,
                            "RATE_DOWN_PAYMENT_MEAN": 0.0500036461891566,
                            "IS_REVOLVING_LOAN": 0.0,
                            "Dependents_Index": 0.0,
                            "Income_to_Dependents_Ratio": 360000.0,
                            "Debt_to_Income_Ratio": 2.21875,
                            "Debt_Repayment_Capability_Index": 0.0407094999999999,
                            "Credit_Utilization_Ratio": 0.3477245590509764,
                        },
                    ]
                }
            ]
        }
    }
