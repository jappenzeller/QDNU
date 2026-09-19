SELECT
  subject,
  channels,
  window_s,
  encoding,
  polarity_sign,
  polarity_strength,
  raw_auc,
  oracle_cal_auc
FROM aws004.polarity_results
WHERE prompt_id = 'IBM-HERON'
ORDER BY subject, channels, window_s
