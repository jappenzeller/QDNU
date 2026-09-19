# AWS-004 Data Layer Cost Log

Append-only. One row per cost-incurring action (S3 PUT, Athena scan, etc.).

| timestamp | action | resource | quantity | unit | cost_usd |
|---|---|---|---|---|---|
| 2026-05-09T13:36:41+00:00 | s3_put | s3://qdnu-braket-151186773465-us-east-1/processed/ | 2 | objects | 0.000010 |
| 2026-05-09T13:36:50+00:00 | athena_scan | qdnu-aws004/71491cb6-0170-4da2-a9ad-25d5c3e98c8e | 0 | bytes | 0.000048 |
| 2026-05-09T13:37:13+00:00 | s3_put | s3://qdnu-braket-151186773465-us-east-1/processed/ | 2 | objects | 0.000010 |
| 2026-05-09T13:37:24+00:00 | athena_scan | qdnu-aws004/5271b066-1db7-469f-bb92-a847446bf629 | 1.28701e+06 | bytes | 0.000048 |
| 2026-05-09T16:24:26+00:00 | quicksight_data_source_create | qdnu-athena | 1 | ds | 0.000000 |
| 2026-05-09T16:25:51+00:00 | quicksight_dataset_create | qdnu-braket-results | 1 | ds | 0.000000 |
| 2026-05-09T16:55:27+00:00 | athena_scan | qdnu-aws004/11fd7a24-e472-495d-86c5-9327f1a67ee2 | 1.28701e+06 | bytes | 0.000048 |
| 2026-05-09T16:57:05+00:00 | athena_scan | qdnu-aws004/6fbd7272-6237-424f-a0e9-2222b93ab337 | 1.28701e+06 | bytes | 0.000048 |
| 2026-05-10T12:15:12+00:00 | ibm_heron_ingest | s3://qdnu-braket-151186773465-us-east-1/polarity/ | 28 | put_objects | 0.000140 |
| 2026-05-10T12:15:28+00:00 | athena_scan | qdnu-aws004/35ce795c-871e-4bf9-a602-4709e2892334 | 10212 | bytes | 0.000048 |
| 2026-05-10T12:15:37+00:00 | athena_scan | qdnu-aws004/6a6c1721-47c7-49c2-b55e-58e29de4a375 | 1.28701e+06 | bytes | 0.000048 |
| 2026-05-10T12:23:55+00:00 | athena_scan | qdnu-aws004/87f8c269-bb95-408c-aaae-513b9393001d | 10212 | bytes | 0.000048 |
| 2026-05-10T13:10:20+00:00 | athena_scan | qdnu-aws004/20881f1b-b3b5-4150-9f31-5e1e1e5b3577 | 10212 | bytes | 0.000048 |
| 2026-05-10T13:13:48+00:00 | quicksight_dataset_create | qdnu-polarity-results | 1 | ds | 0.000000 |
