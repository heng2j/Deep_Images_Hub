-- Queries

-- Requester queries


-- Check if all requestion labels exisits 


SELECT label_name 
FROM labels
WHERE label_name IN
(SELECT l.label_name 
FROM labels l
WHERE l.label_name
in ('Apple', 'protein_bar','Banana'));	
	
	
	
SELECT label_name , image_count
FROM labels
WHERE image_count < 100 AND image_count > 10;

	
-- Check if all labels has enough images to train : threshold >= 100
SELECT label_name
FROM labels
WHERE label_name in ('Apple', 'Banana','protein_bar')
AND image_count < 50;


	
-- Select Sample Images
	
SELECT *
FROM images
LIMIT 10;
	
	
SELECT *
FROM images
WHERE label_name = 'Coconut' ;
LIMIT 10;
	

-- Select Sample labels
	
SELECT *
FROM labels
LIMIT 10;
	
	

-- Compose a list of images for training 
SELECT label_name
FROM labels
WHERE label_name in ('Table', 'Chair','Drawer')
AND image_count > 0;
	
	
SELECT label_name, image_count
FROM labels
WHERE image_count > 100 AND image_count < 500;
	
	
SELECT full_hadoop_path, label_name
FROM images
WHERE label_name in ('Table', 'Chair','Drawer');

	
	
-- Select 100 most recent image batches are in New York
	
SELECT ib.batch_id, ib.place_id, ib.submitted_count, ib.on_board_date, pl.city, pl.neighbourhood, pl.geometry 
FROM images_batches AS ib
JOIN places as pl
ON ib.place_id = pl.place_id 
WHERE pl.city = 'NYC'
ORDER BY ib.on_board_date  DESC
LIMIT 100;

	
-- Select all images from the label's children
	
	SELECT full_hadoop_path, image_thumbnail_object_key,label_name
	FROM images
	WHERE label_name IN (SELECT unnest(l.children)
	FROM labels l
	WHERE l.label_name = 'Food');

	
	
-- Insert New Label with given parent Label
-- When insert, it should insert into the children array of the parent label and all the way up to Entity 
								  
INSERT INTO labels (label_name, parent_name, children, image_count, updated_date) 
VALUES
('LaCroix_Sparkling_Water_Lemon', 'Soft_drink', NULL, 0, (SELECT NOW()));

WITH RECURSIVE labeltree AS ( 
	 SELECT parent_name
     FROM labels 
     WHERE label_name = 'LaCroix_Sparkling_Water_Lemon'
     UNION ALL
     SELECT l.parent_name 
     FROM labels l 
     INNER JOIN labeltree ltree 
	 ON ltree.parent_name = l.label_name 
     WHERE l.parent_name IS NOT NULL) 
		
	 UPDATE labels 
	 SET children = array_append(children, 'LaCroix_Sparkling_Water_Lemon') 
	 FROM labeltree	ltree										  
	 WHERE label_name = ltree.parent_name;
                
-- Verify the previous updates
SELECT *
FROM labels
WHERE label_name = 'Entity';
	
SELECT *
FROM labels
WHERE label_name = 'Drink';
	
SELECT *
FROM labels
WHERE label_name = 'Soft_drink';													  
														  


	
	
-- Insert into requesting_label_watchlist
DELETE FROM requesting_label_watchlist;
INSERT INTO requesting_label_watchlist  (label_name, user_ids,last_requested_userid, new_requested_date ) VALUES

( 'Apple',ARRAY[1], 1, (SELECT NOW()) );
	
	
INSERT INTO requesting_label_watchlist (label_name, user_ids,last_requested_userid, new_requested_date ) VALUES

( 'Apple',ARRAY[1], 1, (SELECT NOW()) )
 ON CONFLICT (label_name)  
 DO
 UPDATE
 SET user_ids = array_append(requesting_label_watchlist.user_ids, 1),
 	 last_requested_userid = 1
 
 WHERE requesting_label_watchlist.label_name = 'Apple';
 


SELECT * 
FROM requesting_label_watchlist;
 
 
 
-- Insert new training request into training_records
DELETE FROM training_records;
ALTER SEQUENCE training_records_model_id_seq RESTART WITH 1;

    INSERT INTO training_records (label_names, image_counts_for_labels, initial_requested_user_id, creation_date ) 
    VALUES 
     
    (ARRAY['Snack', 'Footwear', 'Vehicle'],
     
    (SELECT ARRAY(  
    with x (id_list) as (
      values (ARRAY['Snack', 'Footwear', 'Vehicle'])
    )
    select  image_count
    from labels, x
    where label_name = any (x.id_list)
    order by array_position(x.id_list, label_name)
    )
    )
     ,2, (SELECT NOW()))
     
    RETURNING model_id;

		
	 
SELECT ARRAY(  
with x (id_list) as (
  values (array['Apple','Drawer'])
)
select  image_count
from labels, x
where label_name = any (x.id_list)
order by array_position(x.id_list, label_name)
);
										  
										  
										  
SELECT ARRAY(
SELECT image_count
FROM labels
WHERE label_name IN ('Table','Drawer','Chair')
);									  
										  
SELECT label_name, image_count
FROM labels
WHERE label_name IN ('Apple','Drawer');										  
										  
										  
										  
										  
-- Update training_records with training results
UPDATE training_records
SET final_accuracy = 0.8775,
 	final_validation_accuracy = 0.5643,
	final_loss = 0.0394,
	final_validation_loss = 0.0765,								  
    saved_model_path = 'A path in in S3',
	creation_date	= 	(SELECT NOW())			  
WHERE model_id = 1;								  
										  
						
										  
select * from 	training_records;		
						 
						 

select label_name from labels;