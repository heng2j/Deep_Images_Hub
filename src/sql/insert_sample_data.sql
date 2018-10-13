-- Sample Data

-- Inserts
-- Insert Sample data into labels table
DELETE FROM labels;

INSERT INTO labels (label_name, parent_name, children,image_count,updated_date) VALUES
    ('Think_thin_high_protein_caramel_fudge', 'protein_bar', NULL , 0,(SELECT NOW())),
    ('protein_bar', 'packaged_food', '{"Think_thin_high_protein_caramel_fudge"}', 0,(SELECT NOW())),
    ('packaged_food', 'food', '{"protein_bar","snack"}'  , 0,(SELECT NOW()) ),
	('Apple', 'fruit', NULL, 0 ,(SELECT NOW())),
	('fruit', 'food', '{"Apple","Banana"}' , 0 ,(SELECT NOW()) ),
	('food', 'Entity', '{"packaged_food","fruit"}', 0 ,(SELECT NOW())),
	('furniture', 'equipment', '{"chair","diningtable","sofa"}', 0 ,(SELECT NOW())),
	('equipment', 'Entity', '{"container","furniture"}', 0 ,(SELECT NOW())),
	('animal', 'Entity', '{"bird", "cat","dog","cow","horse","sheep"}' , 0,(SELECT NOW())),
	('transportation', 'Entity', '{"train", "motorbike","car","bus","boat","bicylce", "aeroplane"}', 0 ,(SELECT NOW())),
	('Entity', null,  '{"food", "animal", "equipment", "person", "transportation"}' , 0 ,(SELECT NOW())  );
						
SELECT * FROM labels;


-- Insert Sample data into labels table
INSERT INTO labels (label_name, parent_name, children, image_count, updated_date) VALUES
	('Banana', 'fruit', NULL, 120, Null   );
						
SELECT * FROM labels;




-- Append entity "footware" into the children array of 'Entity'
UPDATE labels set children = array_append(children, 'footware') where label_name = 'Entity';
						
SELECT * FROM labels;




-- Insert image info into users table
DELETE FROM users;
ALTER SEQUENCE users_user_id_seq RESTART WITH 1;

INSERT INTO users (user_name, user_email, user_role, on_board_date) VALUES

	( 'supplier_1', 'heng2j@gmail.com', 'supplier', (SELECT NOW()) ),
	( 'customer_1', 'heng2j@gmail.com', 'customer', (SELECT NOW()) ) ;

SELECT * FROM users;

	 

-- Insert new place info into places table and return newly created place id
DELETE FROM places ;
ALTER SEQUENCE places_place_id_seq RESTART WITH 1;

INSERT INTO places (place_id, licence, postcode, neighbourhood, city, country, lon, lat, geometry, time_added ) VALUES

	( 1 , 'UNKNOWN', 10038, 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 0, 0 , NULL, (SELECT NOW()) )
	 ON CONFLICT (place_id)  DO NOTHING RETURNING place_id ;


INSERT  INTO places(place_id, licence, postcode, neighbourhood, city, country, lon, lat, geometry, time_added)
VALUES
(141511485, 'Data © OpenStreetMap contributors, ODbL 1.0. https://osm.org/copyright', 11377, 'Woodside', 'NYC', 'USA', -73.90173357686848, 40.745229792203425, NULL, (SELECT NOW()) )
ON CONFLICT(place_id)  DO NOTHING RETURNING place_id;

SELECT * FROM places;
	
	 
	 
	 
-- Insert images_batches info into batches table
DELETE FROM images_batches;
ALTER SEQUENCE images_batches_batch_id_seq RESTART WITH 1;


INSERT INTO images_batches (user_id, ready, place_id, submitted_count , on_board_date ) VALUES

	( 1 ,False, 1, 500, (SELECT NOW()) ) RETURNING batch_id;

SELECT * FROM images_batches;


	 
	 
-- Insert image info into images table
DELETE FROM images;
ALTER SEQUENCE images_image_id_seq RESTART WITH 1;
INSERT INTO images (image_object_key, bucket_name, parent_labels,label_name,batch_id,submission_time,user_id,place_id,geometry,image_index,embeddings) VALUES

	('image_object_key', 'bucket_name', 'parent_labels','label_name',1, (SELECT NOW()),1, 1,NULL,NULL,NULL );

	 
SELECT * FROM images i;
	 
SELECT * 
FROM images i
JOIN places pl 
ON i.place_id = pl.place_id;
	 
	 
	 