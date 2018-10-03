-- Queries for customers from Web UI


-- Select 9 most recent image batches 
SELECT (SELECT label_name FROM images WHERE batch_id in (ib.batch_id) LIMIT 1 ) AS label_name , ib.submitted_count, ib.on_board_date, pl.city, pl.neighbourhood, pl.geometry, (SELECT image_thumbnail_object_key FROM images WHERE batch_id in (ib.batch_id)ORDER BY random() LIMIT 1 ) AS sample_image
FROM images_batches AS ib
INNER JOIN places as pl
ON ib.place_id = pl.place_id
WHERE ib.submitted_count > 0
ORDER BY ib.on_board_date  DESC
LIMIT 9;



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
SELECT l.label_name, l.parent_name, l.children, l.image_count, l.updated_date, (SELECT image_thumbnail_object_key FROM images WHERE label_name in (l.label_name) ORDER BY random() LIMIT 1 ) AS sample_image
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
	
