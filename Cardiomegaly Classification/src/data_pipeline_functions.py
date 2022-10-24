'''  This file contains the functions for the data pipeline
'''

import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import List, Dict

from .utils.pandas_utils import explode, create_pivot, filter_df_isin

pd.options.mode.chained_assignment = None  # default='warn' #copy slice warning
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 20})


def dfCleaning(df: pd.DataFrame) -> pd.DataFrame:
    '''Apply Mimic specific replacement functions to dataframe

    :param df: Datframe to adjust
    :type df: pd.DataFrame
    :return: Adjusted dataframe
    :rtype: pd.DataFrame
    '''
    
    # Map Positive and negative values to 0 for negative and 1 for positive
    df.replace({'Negative': 0, 'Positive': 1}, inplace=True)

    # Remove the .dicom from the file paths and replace with .jpg
    df['path'] = df.apply(lambda row: os.path.splitext(row['path'])[0], axis=1)
    df['path'] = df['path'].astype(str) + '.jpg'

    # Replace unrealistic CTR and CPAR values with NaN
    df.loc[df.CTR >= 1, 'CTR'] = np.nan
    df.loc[df.CPAR >= 1, 'CPAR'] = np.nan

    return df


def x_ray_dataframe_generator(
    label: str,
    view: str,
    df_cxr_records: pd.DataFrame,
    df_nb: pd.DataFrame,
    df_cx: pd.DataFrame,
    df_cxr_meta_data: pd.DataFrame,
    df_split: pd.DataFrame,
) -> pd.DataFrame:
    '''Generating a dataframe containing X-ray studies which are available in MIMIC-CXR by merging and filtering

    :param label: Disease column to keep
    :type label: str
    :param view: X-Ray view to filter for
    :type view: str
    :param df_cxr_records: Dataframe with paths for each CXR file
    :type df_cxr_records: pd.DataFrame
    :param df_nb: Dataframe with negbio labels
    :type df_nb: pd.DataFrame
    :param df_cx: Dataframe with chexpert labels
    :type df_cx: pd.DataFrame
    :param df_cxr_meta_data: Dataframe containing meta data info
    :type df_cxr_meta_data: pd.DataFrame
    :param df_split: Dataframe containing info about test and train split
    :type df_split: pd.DataFrame
    :return: Merged and filtered dataframe
    :rtype: pd.DataFrame
    '''

    # Prep cheexpert and negbio
    ## Merge chexpert with negbio
    df_cxnb = df_nb.merge(
        df_cx.drop('subject_id', axis=1),
        how='left',
        left_on='study_id',
        right_on='study_id',
        suffixes=('', '_cx'),
    )
    ## Subselect to training set
    study_ids_train = set(df_split.loc[df_split['split'] == 'train', 'study_id'])
    df_cxnb_train = df_cxnb.loc[df_cxnb['study_id'].isin(study_ids_train)]
    ## Label disagreements in label
    c_cx = f'{label}_cx'
    idx1 = df_cxnb_train[label].isnull() & df_cxnb_train[c_cx].notnull()
    idx2 = df_cxnb_train[label].notnull() & df_cxnb_train[c_cx].isnull()
    idx3 = (
        df_cxnb_train[label].notnull()
        & df_cxnb_train[c_cx].notnull()
        & (df_cxnb_train[label] != df_cxnb_train[c_cx])
    )
    df_cxnb_train.loc[(idx1 | idx2 | idx3), label] = -9
    df_cxnb_train_red = df_cxnb_train[['subject_id', 'study_id', label]]
    ## Annotate
    labels = {0: 'Negative', 1: 'Positive', -1: 'Uncertain', -9: 'Disagreement'}
    df_cxnb_train_red[label] = df_cxnb_train_red[label].map(labels)

    # Prep meta and records
    ## Merge meta with records
    df_meta_records = df_cxr_meta_data[
        [
            'dicom_id',
            'subject_id',
            'study_id',
            'StudyDate',
            'StudyTime',
            'ViewPosition',
        ]
    ].merge(
        df_cxr_records.drop(['study_id', 'subject_id'], axis=1),
        on='dicom_id',
        how='inner',
    )

    # Merge dataframes
    df_combined = df_meta_records.merge(
        df_cxnb_train_red.drop(['subject_id'], axis=1), on='study_id', how='inner'
    )

    # Filter to only include the +/- PA x-rays and for specific views
    df_combined_filtered = df_combined.loc[
        df_combined[label].isin(
            [
                'Positive',
                'Negative',
            ]
        )
    ]
    df_combined_filtered = df_combined_filtered.loc[
        df_combined_filtered['ViewPosition'] == view
    ]

    return df_combined_filtered


def filter_pd_read_chunkwise(
    file_path: str,
    filter_col: str,
    filter_list: List[str],
    chunksize: float,
    dtype: dict = None,
) -> pd.DataFrame:
    '''Use pd.read_csv to read csv file in chunks and filter rows where filter_col is in filter_list

    :param file_path: File path
    :type file_path: str
    :param filter_col: Column to filter by
    :type filter_col: str
    :param filter_list: Values to keep
    :type filter_list: List[str]
    :param chunksize: Chunk size
    :type chunksize: float
    :return: Filtered dataframe
    :rtype: pd.DataFrame
    '''

    chunk_csv = pd.read_csv(file_path, chunksize=chunksize, dtype=dtype)
    filtered_df = pd.concat(
        [filter_df_isin(chunk, filter_col, filter_list) for chunk in chunk_csv]
    )

    return filtered_df


def icu_xray_matcher(
    label: str,
    days_before_icu: int,
    xray_gap_after_icu: int,
    xray_max_time_after_icu: int,
    df_xray: pd.DataFrame,
    df_icu_stays: pd.DataFrame,
) -> pd.DataFrame:
    '''Linking ICU stays to X-ray studies based on their dates
    X-rays linked to ICU stays if they occurs up to 'days_before_icu' before admission, or
    between 'xray_gap_after_icu' and 'xray_max_time_after_icu' days after ICU discharge

    :param label: Disease column
    :type label: str
    :param days_before_icu: Days before icu stay to consider for xray
    :type days_before_icu: int
    :param xray_gap_after_icu: Waiting period after ICU for xrays to consider
    :type xray_gap_after_icu: int
    :param xray_max_time_after_icu: Maximum days after icu stay to consider for xray
    :type xray_max_time_after_icu: int
    :param df_xray: Dataframe with xrays
    :type df_xray: pd.DataFrame
    :param df_icu_stays: Dataframe with icu stays
    :type df_icu_stays: pd.DataFrame
    :return: Combined dataframe
    :rtype: pd.DataFrame
    '''

    # Filter for patients in icu stays with xrays
    unique_full_patients = (
        pd.merge(df_xray, df_icu_stays, on=['subject_id'], how='inner')
        .drop_duplicates(subset=['subject_id'])['subject_id']
        .reset_index(drop=True)
        .tolist()
    )
    cxrOverlap = df_xray.loc[df_xray['subject_id'].isin(unique_full_patients), :]
    icuOverlap = df_icu_stays.loc[
        df_icu_stays['subject_id'].isin(unique_full_patients), :
    ]

    # Convert objects to date times
    icuOverlap['intime'] = pd.to_datetime(icuOverlap['intime'])
    icuOverlap['outtime'] = pd.to_datetime(icuOverlap['outtime'])

    # The matching works like this:
    # First we iterate through the patients who have both ICU stays and X-ray studies.
    # For each patient we create lists for the dates, study_ids, labels, paths and view positions for all of their X-rays in MIMIC
    ICUMatcher = icuOverlap[['stay_id', 'intime', 'outtime']].copy()
    ICUMatcher['Match'] = 0
    ICUMatcher['study_id'] = 0
    ICUMatcher['Label'] = 0
    ICUMatcher['ViewPosition'] = 0
    ICUMatcher['path'] = 0
    ICUMatcher['EarlyBoundary'] = ICUMatcher['intime'] - datetime.timedelta(
        days=days_before_icu
    )
    ICUMatcher['PostGapStart'] = ICUMatcher['outtime'] + datetime.timedelta(
        days=xray_gap_after_icu
    )
    ICUMatcher['PostGapStop'] = ICUMatcher['PostGapStart'] + datetime.timedelta(
        days=xray_max_time_after_icu
    )
    # Iterate through all of the subjects
    for subid in unique_full_patients:
        PatientCXR = cxrOverlap.loc[cxrOverlap['subject_id'] == subid].reset_index(
            drop=True
        )
        PatientICU = icuOverlap.loc[icuOverlap['subject_id'] == subid].reset_index(
            drop=True
        )

        # Iterate through CXR
        CXRdates = []
        CXRstudies = []
        CXRlabels = []
        CXRpaths = []
        CXRview = []
        for (
            _,
            row,
        ) in (
            PatientCXR.iterrows()
        ):  # Populate lists with dates, times and labels of each CXR study
            date_temp = str(row['StudyDate'])
            time_temp = str(row['StudyTime']).split('.')
            if len(time_temp[0]) < 6:
                time_temp[0] = '0' * (6 - len(time_temp[0])) + time_temp[0]
            time_temp = '.'.join(time_temp)
            time_aux = 'T'.join([date_temp, time_temp])
            studytime = pd.to_datetime(time_aux)
            CXRdates.append(
                studytime
            )  # Add time for this study to the list define above
            CXRstudies.append(row['study_id'])  # Add study
            CXRlabels.append(row[label])  # Add label to label list
            CXRpaths.append(row['path'])  # Add path to list of paths
            CXRview.append(row['ViewPosition'])  # Add view position to list of views

        # Iterate through the ICU Stays
        for (
            _,
            row,
        ) in PatientICU.iterrows():  # For each of the patient's ICU stays
            ICUMatcherRow = ICUMatcher.loc[
                ICUMatcher.stay_id == row['stay_id']
            ]  # First get the stay ID
            CXRs_in_range = (
                []
            )  # Initialise an empty list which will record which of the patient's CXR's are in range of
            # this ICU stay

            for i in range(
                len(CXRdates)
            ):  # Iterate through the CXR dates and check if any fall inside the time
                # window we defined for the ICU-Stay (Daysbefore, daysafter etc)
                Date = CXRdates[i]  # First get the date of the CXR study

                # Then check if that date falls in the specified range
                if (
                    (
                        (Date > ICUMatcherRow['EarlyBoundary']).bool()
                        and (Date < ICUMatcherRow['intime']).bool()
                    )
                    or (
                        (Date > ICUMatcherRow['intime']).bool()
                        and (Date < ICUMatcherRow['outtime']).bool()
                    )
                    or (
                        (Date > ICUMatcherRow['PostGapStart']).bool()
                        and (Date < ICUMatcherRow['PostGapStop']).bool()
                    )
                ):

                    CXRs_in_range.append(
                        i
                    )  # If an X-ray falls in the specified window, we add it's index to the list
                    # that we initialised earlier

            # Now we know which id's from the cxr list fall in the range
            # Filter down the earlier lists so they only contain CXRs within the ICU range
            CXRdates_in_range = [
                CXRdates[i] for i in CXRs_in_range
            ]  # Just the dates of the X-rays which fell in the range
            CXRlabels_in_range = [
                CXRlabels[i] for i in CXRs_in_range
            ]  # The labels of the X-rays which fell in the range

            if len(CXRlabels_in_range) > 0:  # If at least one X-ray in the range

                # Need to find which X-ray in the range was taken closest to ICU admission.
                DaysAway = (
                    []
                )  # This list will store the distance from the ICU in-time to each X-ray study date
                # and let us pick the nearest date

                # Issues often crop up when doing this due to datatype issues but this should work.
                for x in range(len(CXRdates_in_range)):
                    TempDates = abs(
                        CXRdates_in_range[x] - ICUMatcherRow['intime']
                    )  # Get the absolute values of time between
                    # X-ray and ICU admission
                    TempDates = TempDates.astype(
                        'timedelta64[D]'
                    )  # need to convert the datatype
                    DaysAway.append(
                        TempDates.astype(int).values
                    )  # Add the value to the list

                TempIndex = DaysAway.index(
                    min(DaysAway)
                )  # Once the DaysAway list is populated we can get the index
                # of the minimum value

                ChosenDate = CXRdates_in_range[
                    TempIndex
                ]  # Using the index we can get the date from the CXRdates_in_range list
                NearestIndex = CXRdates.index(
                    ChosenDate
                )  # Using the date we can then get the index of that study in the earlier
                # CXRdates list

                # If we find a X-ray does fall window the predefined ICU time window, set flags in the matcher dataframes.
                # Then we copy the x-ray label into the ICU matcher dataframe along with the X-ray's study_id
                ICUMatcher.loc[
                    ICUMatcher.stay_id == row['stay_id'], 'Match'
                ] = 1  # set the matcher flag
                ICUMatcher.loc[
                    ICUMatcher.stay_id == row['stay_id'], 'study_id'
                ] = CXRstudies[
                    NearestIndex
                ]  # copy the study id
                # of the nearest x-ray into the ICUMatcher dataframe

                # Copy the X-ray's label, view position and path into the same dataframe
                ICUMatcher.loc[ICUMatcher.stay_id == row['stay_id'], label] = CXRlabels[
                    NearestIndex
                ]
                ICUMatcher.loc[
                    ICUMatcher.stay_id == row['stay_id'], 'ViewPosition'
                ] = CXRview[NearestIndex]
                ICUMatcher.loc[ICUMatcher.stay_id == row['stay_id'], 'path'] = CXRpaths[
                    NearestIndex
                ]

    # Getting rid of non-matches
    ICUMatcher = ICUMatcher.loc[ICUMatcher['Match'] == 1].reset_index(drop=True)
    df_combined = icuOverlap.merge(
        ICUMatcher.drop(['intime', 'outtime'], axis=1), how='right', on='stay_id'
    )

    return df_combined


def SignalTableGenerator(
    df_icu_xray: pd.DataFrame,
    df_icu_timeseries: pd.DataFrame,
    df_icu_lab: pd.DataFrame,
    df_patients: pd.DataFrame,
    df_admissions: pd.DataFrame,
    df_ctr: pd.DataFrame,
    df_cpar: pd.DataFrame,
    chart_labels_mean: Dict[int, str],
    chart_labels_max: Dict[int, str],
    chart_labels_min: Dict[int, str],
    lab_labels_mean: Dict[int, str],
    lab_labels_max: Dict[int, str],
    lab_labels_min: Dict[int, str],
    average_by: str,
) -> pd.DataFrame:
    '''Average ICU data (vitals and labs) over a stay (or hour) and add it to the dataframe
    with the linked X-ray study together with patient and admission data, plus image-extracted 
    biomarker values

    :param df_icu_xray: DataFrame to merge on
    :type df_icu_xray: pd.DataFrame
    :param df_icu_timeseries: ICU timeseries data
    :type df_icu_timeseries: pd.DataFrame
    :param df_icu_lab: Lab timeseries data
    :type df_icu_lab: pd.DataFrame
    :param df_patients: Patient info data
    :type df_patients: pd.DataFrame
    :param df_admissions: Admission data
    :type df_admissions: pd.DataFrame
    :param df_ctr: CTR values
    :type df_ctr: pd.DataFrame
    :param df_cpar: CPAR values
    :type df_cpar: pd.DataFrame
    :param chart_labels_mean: Labels
    :type chart_labels_mean: Dict[int, str]
    :param chart_labels_max: Labels
    :type chart_labels_max: Dict[int, str]
    :param chart_labels_min: Labels
    :type chart_labels_min: Dict[int, str]
    :param lab_labels_mean: Labels
    :type lab_labels_mean: Dict[int, str]
    :param lab_labels_max: Labels
    :type lab_labels_max: Dict[int, str]
    :param lab_labels_min: Labels
    :type lab_labels_min: Dict[int, str]
    :param average_by: Average bu Hourly or Stay
    :type average_by: str
    :return: Combined dataframe
    :rtype: pd.DataFrame:
    '''

    # Prep df_icu_xray dataframe
    df_icu_xray = df_icu_xray.drop_duplicates(subset=['path'])
    df_icu_xray['intime'] = pd.to_datetime(df_icu_xray['intime'])
    df_icu_xray['outtime'] = pd.to_datetime(df_icu_xray['outtime'])

    # Merge the patient info and admission table
    df_patient_admission = (
        df_admissions[['subject_id', 'ethnicity']]
        .drop_duplicates(subset=['subject_id'])
        .merge(
            df_patients[['subject_id', 'anchor_age', 'anchor_year', 'gender']].drop_duplicates(
                subset=['subject_id']
            ),
            on='subject_id',
            how='left',
        )
    )
    
    # Merge df_patient_admission onto df_icu_xray based on subject_id
    df_icu_xray_patient_admission = df_icu_xray.merge(
        df_patient_admission, how='left', on='subject_id'
    )

    # Compare year intime to anchor_year to find approximate admission age --> kept in 'anchor_age' column to avoid future conflicts
    df_icu_xray_patient_admission['anchor_age'] = df_icu_xray_patient_admission['anchor_age'] + df_icu_xray_patient_admission['intime'].dt.year - df_icu_xray_patient_admission['anchor_year']
        
    df_icu_xray_patient_admission.drop(['anchor_year'],axis=1)
    

    # Prep timeseries tables

    if average_by == 'Hourly':
        df_icu_xray_patient_admission = explode_icu_stays(df_icu_xray_patient_admission)

    ## Adjust timestamps
    if average_by == 'Hourly':
        df_icu_timeseries['charttime'] = pd.to_datetime(df_icu_timeseries['charttime'])
        df_icu_lab['charttime'] = pd.to_datetime(df_icu_lab['charttime'])
        df_icu_timeseries['charttime'] = df_icu_timeseries['charttime'].dt.round(
            '60min'
        )
        df_icu_lab['charttime'] = df_icu_lab['charttime'].dt.round('60min')

    ## Get matchers
    if average_by == 'Hourly':
        ChartUniqueMatcher = ['subject_id', 'charttime']
        LabUniqueMatcher = ['subject_id', 'charttime']
    elif average_by == 'Stay':
        ChartUniqueMatcher = 'stay_id'
        LabUniqueMatcher = 'hadm_id'

    ## Create pivot tables from timeseries table for various aggregation levels and merge together
    df_icu_timeseries_pivoted = create_pivot(
        df=df_icu_timeseries,
        labels=chart_labels_mean,
        labels_column='itemid',
        unique_matcher=ChartUniqueMatcher,
        aggfunc='mean',
        values='valuenum',
    )
    pivot_aux = create_pivot(
        df=df_icu_timeseries,
        labels=chart_labels_max,
        labels_column='itemid',
        unique_matcher=ChartUniqueMatcher,
        aggfunc='max',
        values='valuenum',
    )
    df_icu_timeseries_pivoted = df_icu_timeseries_pivoted.merge(
        pivot_aux, how='outer', on=ChartUniqueMatcher
    )
    pivot_aux = create_pivot(
        df=df_icu_timeseries,
        labels=chart_labels_min,
        labels_column='itemid',
        unique_matcher=ChartUniqueMatcher,
        aggfunc='min',
        values='valuenum',
    )
    df_icu_timeseries_pivoted = df_icu_timeseries_pivoted.merge(
        pivot_aux, how='outer', on=ChartUniqueMatcher
    )
    df_icu_lab_pivoted = create_pivot(
        df=df_icu_lab,
        labels=lab_labels_mean,
        labels_column='itemid',
        unique_matcher=LabUniqueMatcher,
        aggfunc='mean',
        values='valuenum',
    )
    pivot_aux = create_pivot(
        df=df_icu_lab,
        labels=lab_labels_max,
        labels_column='itemid',
        unique_matcher=LabUniqueMatcher,
        aggfunc='max',
        values='valuenum',
    )
    df_icu_lab_pivoted = df_icu_lab_pivoted.merge(
        pivot_aux, how='outer', on=LabUniqueMatcher
    )
    pivot_aux = create_pivot(
        df=df_icu_lab,
        labels=lab_labels_min,
        labels_column='itemid',
        unique_matcher=LabUniqueMatcher,
        aggfunc='min',
        values='valuenum',
    )
    df_icu_lab_pivoted = df_icu_lab_pivoted.merge(
        pivot_aux, how='outer', on=LabUniqueMatcher
    )

    # Merge the timeseries pivot tables onto the df_icu_xray_patient_admission table
    df_icu_xray_patient_admission_timeseries = df_icu_xray_patient_admission.merge(
        df_icu_timeseries_pivoted, how='left', on=ChartUniqueMatcher
    )
    df_icu_xray_patient_admission_timeseries_lab = (
        df_icu_xray_patient_admission_timeseries.merge(
            df_icu_lab_pivoted, how='left', on=LabUniqueMatcher
        )
    )

    # Merge CTR and CPAR dataframes onto df_icu_xray_patient_admission_timeseries_lab table
    df_icu_xray_patient_admission_timeseries_lab['dicom_file'] = pd.Series(df_icu_xray_patient_admission_timeseries_lab.path.str[-48:-4])

    df_icu_xray_patient_admission_timeseries_lab_ctr = df_icu_xray_patient_admission_timeseries_lab.merge(df_ctr, on = 'dicom_file')
    df_icu_xray_patient_admission_timeseries_lab_ctr_cpar = df_icu_xray_patient_admission_timeseries_lab_ctr.merge(df_cpar, on = 'dicom_file')

    df_icu_xray_patient_admission_timeseries_lab_ctr_cpar.drop(labels='dicom_file', axis=1, inplace=True)

    # return final output
    return df_icu_xray_patient_admission_timeseries_lab_ctr_cpar


def explode_icu_stays(df: pd.DataFrame) -> pd.DataFrame:
    '''Create a new column 'charttime' in df
    For each row in df:
    Get intime and outime of that ICU stay
    Create a list of date-times in hourly increments between intime and outime
    (For example if someone was in the ICU on 05/12/2020 from 2pm till 5pm, create the list [2020-12-05 14:00:000,
    2020-12-05 15:00:000, 2020-12-05 16:00:000,])
    Insert this list into the 'charttime' column

    After inserting this list into every row of df, we then explode the dataframe based on the charttime column
    Each value in the charttime list for that row will now get its own row.

    All of the values in the new row (i.e. subject id, patient age, study id...) will be the same as in the original row
    except that the charttime will be different for each row

    The example above would produce 3 rows, each with a different charttime but all the other
    values (id, age, ethnicity, etc) would be the same

    :param df: Dataframe to explode
    :type df: pd.DataFrame
    :return: Exploded dataframe
    :rtype: pd.DataFrame
    '''

    # insert charttime column
    df.insert(
        1, 'charttime', np.nan
    )  # Chart time will be the actual date and time, and is
    # used for matching chart events
    df['charttime'] = df['charttime'].astype('object')  # set type as object

    # Also include a Time column which will be an integer for each hour
    df.insert(1, 'Time', np.nan)  # i.e. 1 for hour 1, 2 for the second hour, 3...
    df['Time'] = df['Time'].astype('object')

    # Round intime and outime to the hour, (everything so far has already been rounded to the hour)
    df['intime'] = df['intime'].dt.round('60min')
    df['outtime'] = df['outtime'].dt.round('60min')

    # Iterate through rows in df
    for i, _ in df.iterrows():  # for each row
        datelist = pd.date_range(
            start=df.loc[i, 'intime'],
            end=df.loc[i, 'outtime'],
            freq='60min',
        ).tolist()
        # create the list of datetimes in hourly increments

        timelist = list(range(len(datelist)))  # also create the list of integers
        df.at[i, 'charttime'] = datelist  # insert charttime lists into the row
        df.at[i, 'Time'] = timelist  # also insert the integer time list into the row

    df_exploded = explode(
        df, lst_cols=['charttime', 'Time']
    )  # Explode the dataframe using the times,
    # so that each row now represents an hour but still has the same patient information (age, gender) as when
    # each row represented an entire stay

    return df_exploded
