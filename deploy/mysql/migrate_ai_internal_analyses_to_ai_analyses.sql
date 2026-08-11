INSERT INTO ai_analyses (
    analysis_id, created_at, updated_at, result_json,
    store_id, svc_induty_cd, trdar_cd, user_id, yyqu_cd
)
SELECT
    src.analysis_id,
    STR_TO_DATE(SUBSTRING_INDEX(src.created_at, '+', 1), '%Y-%m-%dT%H:%i:%s.%f') AS created_at,
    STR_TO_DATE(SUBSTRING_INDEX(src.created_at, '+', 1), '%Y-%m-%dT%H:%i:%s.%f') AS updated_at,
    JSON_OBJECT(
        'report', CAST(src.report_json AS JSON),
        'diagnosis', CAST(src.diagnosis_json AS JSON),
        'warnings', CAST(src.warnings_json AS JSON),
        'detailed_analysis', CASE
            WHEN src.detailed_analysis_json IS NULL THEN NULL
            ELSE CAST(src.detailed_analysis_json AS JSON)
        END
    ) AS result_json,
    CASE WHEN src.store_id REGEXP '^[0-9]+$' THEN CAST(src.store_id AS UNSIGNED) ELSE NULL END AS store_id,
    src.svc_induty_cd,
    src.trdar_cd,
    CASE WHEN src.user_id REGEXP '^[0-9]+$' THEN CAST(src.user_id AS UNSIGNED) ELSE NULL END AS user_id,
    src.yyqu_cd
FROM ai_internal_analyses AS src
WHERE src.analysis_id NOT IN (SELECT analysis_id FROM ai_analyses);
