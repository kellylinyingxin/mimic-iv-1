--1
SELECT pa.*
FROM (SELECT a.subject_id,
             a.hadm_id,
             a.admittime,
             p.gender,
             FLOOR((CAST(a.admittime AS INTEGER) - CAST(p.dob AS INTEGER)) / 365.23076923) AS age
      FROM admissions a
      JOIN patients p ON a.subject_id = p.subject_id
     ) pa
JOIN chartevents e ON pa.hadm_id = e.hadm_id AND e.itemid = 226732 AND value = 'Endotracheal tube'
WHERE 1=1
AND pa.age > 18

--2
-- Assumptions:
----------------- a patient may have mechanical ventilation episodes across multiple hospitalizations.
----------------- charttime is the only unique identifier of chartevents.
WITH start_mechanical_ventilation_population AS (
    SELECT pa.subject_id,
           pa.hadm_id,
           pa.admittime,
           pa.gender,
           pa.patientAge,
           ROW_NUMBER() OVER (PARTITION BY pa.subject_id, pa.hadm_id ORDER BY e.charttime) AS startMechanicalVentilation,
           e.charttime
    FROM (SELECT a.subject_id,
                 a.hadm_id,
                 a.admittime,
                 p.gender,
                 FLOOR((CAST(a.admittime AS INTEGER) - CAST(p.dob AS INTEGER)) / 365.23076923) AS patientAge
          FROM admissions a
          JOIN patients p ON a.subject_id = p.subject_id
         ) pa
    JOIN chartevents e ON pa.hadm_id = e.hadm_id AND e.itemid = 226732 AND value = 'Endotracheal tube'
    WHERE 1=1
    AND pa.age > 18
--    GROUP BY pa.subject_id,
--             pa.hadm_id,
--             pa.admittime,
--             pa.gender,
--             pa.patientAge,
--             e.charttime
    )

, all_events_ordered AS (
    SELECT e.*,
           ROW_NUMBER() OVER (PARTITION BY e.subject_id, e.hadm_id ORDER BY e.charttime) AS eventRowNumber
    FROM chartevents e
    WHERE 1=1
    AND e.subject_id IN (SELECT DISTINCT subject_id FROM start_mechanical_ventilation_population)
    )

, end_of_first_mechanical_ventilation_episode AS (
    SELECT e.subject_id,
           e.hadm_id,
           st.charttime AS mechanicalVentilationStartTime,
           MIN(e.eventRowNumber) AS firstEpisodeEndRowNumber
    FROM all_events_ordered e
    JOIN start_mechanical_ventilation_population st ON e.subject_id = st.subject_id AND e.hadm_id = st.hadm_id AND st.startMechanicalVentilation = 1
    WHERE 1=1
    AND e.charttime > st.charttime
    AND e.itemid = 226732
    AND e.value <> 'Endotracheal tube'
    GROUP BY e.subject_id,
             e.hadm_id,
             st.charttime
    )

, testCTE AS (
    SELECT e.hadm_id,
           e.subject_id,
           CASE WHEN e.charttime = mv.mechanicalVentilationStartTime THEN 'start_time'
                WHEN e.eventRowNumber = mv.firstEpisodeEndRowNumber  THEN 'end_time'
           END AS period,
           e.charttime
    FROM all_events_ordered e
    JOIN end_of_first_mechanical_ventilation_episode mv ON e.subject_id = mv.subject_id AND e.hadm_id = mv.hadm_id
    WHERE (e.eventRowNumber = mv.firstEpisodeEndRowNumber-1
           OR
           e.charttime = mv.mechanicalVentilationStartTime
          )
    )
    6


SELECT *
FROM start_mechanical_ventilation_population e
WHERE 1=1
AND e.itemid = 226732
AND e.value = "Endotracheal tube"
AND e.eventRowNumber 


