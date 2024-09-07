import json
import os

import numpy as np
import pandas as pd
import requests
from pandas import DataFrame

# LOAN_COUNT > 0
DEBT_COLS = [
    "AMT_CREDIT_SUM_DEBT_SUM",  # down
    "AMT_CREDIT_SUM_DEBT_MEAN",  # 비례
    "Credit_Utilization_Ratio",  # down
]

INCOME_COLS = [
    "AMT_INCOME_TOTAL",  # up
    "Income_to_Dependents_Ratio",  # 비례
    "Debt_Repayment_Capability_Index",  # down
    "Debt_to_Income_Ratio",  # 반비례
]

EMPLOYMENT_COLS = [
    "DAYS_EMPLOYED",  # down (negative, abs-up)
]


class APIHelper:
    API_URL: str
    API_KEY: str

    def __init__(self, api_url: str, api_key: str):
        self.API_URL = api_url
        self.API_KEY = api_key

    def get_permutations(row: DataFrame, min_maxes: dict, num: int = 100):
        perm_dfs = []
        has_loan_history = row["LOAN_COUNT"].item() > 0

        # 1. 과거 대출 O
        if has_loan_history:
            curr_debt_sum = row["AMT_CREDIT_SUM_DEBT_SUM"].item()
            debt_sums = np.linspace(
                curr_debt_sum,
                min_maxes["AMT_CREDIT_SUM_DEBT_SUM"][0],  # DOWN
                num=num,
            ).tolist()
            debt_means = [(ds / row["LOAN_COUNT"].item()) for ds in debt_sums]
            curs = [
                (
                    0
                    if (row["AMT_CREDIT_SUM"].item() == 0)
                    else (ds / row["AMT_CREDIT_SUM"].item())
                )
                for ds in debt_sums
            ]
            perm_df = pd.concat([row for _ in range(num)], axis=0)
            perm_df["AMT_CREDIT_SUM_DEBT_SUM"] = debt_sums
            perm_df["AMT_CREDIT_SUM_DEBT_MEAN"] = debt_means
            perm_df["Credit_Utilization_Ratio"] = curs
            perm_dfs.append(perm_df)

        # 2. Income
        curr_income_total = row["AMT_INCOME_TOTAL"].item()
        incomes = np.linspace(
            curr_income_total,
            min_maxes["AMT_INCOME_TOTAL"][1],  # UP
            num=num,
        ).tolist()
        income_dpd_ratios = [
            inc / (row["CNT_CHILDREN"].item() + 1) for inc in incomes  # 소득 대비 부양 부담 비율
        ]
        drc_indexes = [
            0 if (inc == 0) else (row["AMT_ANNUITY_MEAN"].item() / inc)
            for inc in incomes
        ]
        dti_ratios = [
            0 if (inc == 0) else (row["AMT_CREDIT_SUM_DEBT_MEAN"].item() / inc)
            for inc in incomes
        ]

        perm_df = pd.concat([row for _ in range(num)], axis=0)
        perm_df["AMT_INCOME_TOTAL"] = incomes
        perm_df["Income_to_Dependents_Ratio"] = income_dpd_ratios
        perm_df["Debt_Repayment_Capability_Index"] = drc_indexes
        perm_df["Debt_to_Income_Ratio"] = dti_ratios
        perm_dfs.append(perm_df)

        # 3. Employment
        curr_days_employed = row["DAYS_EMPLOYED"].item()
        days_employed = np.linspace(
            curr_days_employed,
            min_maxes["DAYS_EMPLOYED"][0],  # DOWN (negative-up)
            num=num,
        )

        perm_df = pd.concat([row for _ in range(num)], axis=0)
        perm_df["DAYS_EMPLOYED"] = days_employed
        perm_dfs.append(perm_df)

        # # 4. Change All at once
        perm_df = pd.concat([row for _ in range(num)], axis=0)
        if has_loan_history:
            perm_df["AMT_CREDIT_SUM_DEBT_SUM"] = debt_sums
            perm_df["AMT_CREDIT_SUM_DEBT_MEAN"] = debt_means
            perm_df["Credit_Utilization_Ratio"] = curs
        perm_df["AMT_INCOME_TOTAL"] = incomes
        perm_df["Income_to_Dependents_Ratio"] = income_dpd_ratios
        perm_df["Debt_Repayment_Capability_Index"] = drc_indexes
        perm_df["Debt_to_Income_Ratio"] = dti_ratios
        perm_df["DAYS_EMPLOYED"] = days_employed
        perm_dfs.append(perm_df)

        result_df = pd.concat([row] + perm_dfs, axis=0)

        # First row is the original
        result_df = result_df.drop_duplicates(keep="first")
        result_df = result_df.reset_index(drop=True)
        return result_df

    def run_permute_inference(self, row: DataFrame, model_name: str = None):
        perm_df = self.get_permutations(row)
        body = json.dumps({"data": perm_df.to_dict(orient="records")})
        result = requests.post(
            f"{self.API_URL}/inference",
            params={"model_name": model_name} if model_name else None,
            data=body,
            headers={"access_token": self.API_KEY},
        ).json()
        result = DataFrame.from_records(result)

        # Post-processing
        result["pred_probs_loan"] = 1 - result["pred_probs"]
        result["preds_loan"] = 1 - result["preds"]

        combined_result = pd.concat(
            [result, perm_df.drop(columns=["SK_ID_CURR", "NAME"])],
            axis=1,
        )
        return combined_result

    def run_inference(self, df: DataFrame, model_name: str = None):
        body = json.dumps({"data": df.to_dict(orient="records")})
        result = requests.post(
            f"{self.API_URL}/inference",
            params={"model_name": model_name} if model_name else None,
            data=body,
            headers={"access_token": self.API_KEY},
        ).json()
        result = DataFrame.from_records(result)

        # Post-processing
        result["pred_probs_loan"] = 1 - result["pred_probs"]
        result["preds_loan"] = 1 - result["preds"]

        combined_result = pd.concat(
            [result, df.drop(columns=["SK_ID_CURR", "NAME"])],
            axis=1,
        )
        return combined_result
