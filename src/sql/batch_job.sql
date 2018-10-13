-- Query for batch job 

-- Get info for un-filtered images for image filtering batch process 

SELECT  ib.batch_id, ib.user_id, ib.place_id
FROM images_batches AS ib
WHERE ib.ready = false;


-- Query Sample
SELECT * FROM images_batches;



