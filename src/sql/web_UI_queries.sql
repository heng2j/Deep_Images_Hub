-- Queries for customers from Web UI


-- Select 9 most recent image batches 
SELECT (SELECT label_name FROM images WHERE batch_id in (ib.batch_id) LIMIT 1 ) AS label_name , ib.submitted_count, ib.on_board_date, pl.city, pl.neighbourhood, pl.geometry, (SELECT image_thumbnail_object_key FROM images WHERE batch_id in (ib.batch_id)ORDER BY random() LIMIT 1 ) AS sample_image
FROM images_batches AS ib
INNER JOIN places as pl
ON ib.place_id = pl.place_id
WHERE ib.submitted_count > 0
ORDER BY ib.on_board_date  DESC
LIMIT 100;

SELECT *
FROM images_batches AS ib
ORDER BY ib.on_board_date  DESC
LIMIT 100;



SELECT * FROM images WHERE label_name = 'Guitar';


select * from images 
ORDER BY submission_time  DESC
limit 10; 



-- Select Max 30 labels order by label_name
SELECT * , (SELECT image_thumbnail_object_key FROM images WHERE label_name in (l.label_name) LIMIT 1 ) AS sample_image
FROM labels AS l
WHERE l.image_count > 0
ORDER BY l.label_name  
LIMIT 30;
	
	
-- Select 100 parent labels order who has children TODO - need to remove l.image_count > 0 for the highest parent
SELECT l.label_name, l.parent_name, l.children, l.image_count, l.updated_date, (SELECT image_thumbnail_object_key FROM images WHERE label_name in (l.label_name) LIMIT 1 ) AS sample_image
FROM labels AS l
WHERE l.image_count > 0
AND  cardinality(l.children) > 0
ORDER BY l.label_name  
LIMIT 100;
	
	
-- Select All labels
SELECT l.label_name, l.parent_name,l.image_count, l.updated_date, (SELECT image_thumbnail_object_key FROM images WHERE label_name in (l.label_name) ORDER BY random() LIMIT 1 ) AS sample_image
FROM labels AS l
WHERE l.image_count > 0
ORDER BY l.label_name ; 


-- Select distinct parent labels within the above above labels
SELECT DISTINCT l.parent_name 
FROM labels AS l
WHERE l.image_count > 0;

	




-- Select 24 labels order by label_name and also filter by parent's Name
SELECT label_name, parent_name,children,image_count,updated_date, (SELECT image_thumbnail_object_key FROM images WHERE label_name in (l.label_name)ORDER BY random() LIMIT 1 ) AS sample_image
FROM labels AS l
WHERE l.image_count > 0
AND l.parent_name = 'Food'
ORDER BY l.label_name  
LIMIT 24;
	
	
select * from images LIMIT 10;


-- Select top 3 most recent trained models with info
-- Will need The Model_id, 
SELECT tr.model_id,tr.label_names,tr.image_counts_for_labels,tr.final_accuracy, tr.final_validation_accuracy, tr.final_loss, tr.final_validation_loss,tr.creation_date, tr.note, array_length(tr.purchased_user_ids,1) AS number_purchased,downloadable_model_link, downloadable_plot_link, (SELECT image_thumbnail_small_object_key FROM images WHERE label_name in (select unnest( tr.label_names) ORDER BY random() LIMIT 1 ) LIMIT 1) AS thumbnail_image
FROM training_records tr
ORDER by tr.creation_date  DESC
LIMIT 3;	


-- Select trained models by model_id
SELECT tr.model_id,tr.label_names,tr.image_counts_for_labels,tr.final_accuracy, tr.final_validation_accuracy, tr.final_loss, tr.final_validation_loss,tr.creation_date, tr.note, array_length(tr.purchased_user_ids,1) AS number_purchased,downloadable_model_link, downloadable_plot_link
FROM training_records tr
WHERE model_id = 17;


SELECT image_thumbnail_small_object_key 
FROM images 
WHERE label_name in (select unnest( tr.label_names) from training_records tr where model_id = 17 ) ;



-- Created get_image_thumbnail_object_keys function to get the sample mage_thumbnail_object_key for each label by model id
DROP FUNCTION get_image_thumbnail_object_keys;
CREATE FUNCTION get_image_thumbnail_object_keys(int) RETURNS SETOF text AS
$BODY$
DECLARE
    temprow   text;
BEGIN
FOR temprow IN
        SELECT unnest( tr.label_names) FROM training_records tr WHERE model_id = $1 
    LOOP
    RETURN QUERY EXECUTE  'SELECT image_thumbnail_object_key FROM images WHERE label_name = ''' || temprow || ''' LIMIT 1' ;
	
	END LOOP;
 END
$BODY$
LANGUAGE plpgsql;

-- Returns a sample image_thumbnail_small_object_key for each label by model id
SELECT * FROM get_image_thumbnail_object_keys(17);



					
					