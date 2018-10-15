-- Create Tables



-- Drop labels table if exists;
-- Create labels table if not exists;
DROP TABLE IF EXISTS labels CASCADE;

CREATE TABLE IF NOT EXISTS labels (
    label_name  text PRIMARY KEY ,
    parent_name text  ,--references labels(label_name),
    children    varchar[],
	image_count	int,
	requested_count int,
	updated_date 		timestamp
);

ALTER TABLE labels
ADD FOREIGN KEY (parent_name)
REFERENCES labels(label_name);

	
SELECT * FROM labels;

-- Drop users table if exists;
-- Create users table if not exists;
DROP TABLE IF EXISTS users CASCADE;

CREATE TABLE IF NOT EXISTS users (
    user_id				SERIAL	PRIMARY KEY ,
	user_name			text,
	user_email			text,
    user_role			text,
	user_organization	text,
	on_board_date 		timestamp

);


-- Drop batches table if exists;
-- Create batches table if not exists;
DROP TABLE IF EXISTS images_batches CASCADE;

CREATE TABLE IF NOT EXISTS images_batches (
    batch_id			SERIAL	PRIMARY KEY ,
	user_id				int references users(user_id),
	ready				boolean,
	place_id			int references places(place_id),
	submitted_count		int,
	accepted_count		int,
	quality_score		float,
	on_board_date 		timestamp
);

SELECT * FROM images_batches;


-- Drop places table if exists;
-- Create places table if not exists;
DROP TABLE IF EXISTS places CASCADE; 

CREATE TABLE IF NOT EXISTS places (
    place_id		int	PRIMARY KEY ,
    licence			text,
    postcode    	int,
	neighbourhood	text,
	city			text,
	country			text,
	lon				float,
	lat 			float,
	geometry 		point,
	time_added		timestamp
);

SELECT * FROM places;
	
	
-- Drop images table if exists;
-- Create images table if not exists;
DROP TABLE IF EXISTS images CASCADE;

CREATE TABLE IF NOT EXISTS images (
    image_id								SERIAL	PRIMARY KEY ,
    image_object_key						text,
	image_thumbnail_object_key 				text,
	bucket_name								text,
	full_hadoop_path						text,
	parent_labels							text, 
	label_name								text,
	batch_id								int references images_batches(batch_id),
	submission_time 						timestamp,
	user_id									int references users(user_id),
	place_id								int references places(place_id),
	image_index								int,
	embeddings								float[],
	verified								boolean
);


-- Altered images to include image_thumbnail_small_object_key column

Alter TABLE images 
ADD COLUMN image_thumbnail_small_object_key text;


SELECT * FROM images LIMIT 10;


-- Drop requests table if exists;
-- Create requests table if not exists;
DROP TABLE IF EXISTS requests;

CREATE TABLE IF NOT EXISTS images_batches (
    request_id			SERIAL	PRIMARY KEY ,
	user_id				int references users(user_id),
	request_date_time 		timestamp,
	labels    varchar[]
);




-- Drop requesting_label_watchlist table if exists;
-- Create requesting_label_watchlist table if not exists;
DROP TABLE IF EXISTS requesting_label_watchlist;

CREATE TABLE IF NOT EXISTS requesting_label_watchlist (
    label_name  text PRIMARY KEY ,
    user_ids    int[],
	last_requested_userid int ,
	new_requested_date 		timestamp
);
	
	
	
-- Drop requesting_label_watchlist table if exists;
-- Create requesting_label_watchlist table if not exists;
DROP TABLE IF EXISTS training_records;

CREATE TABLE IF NOT EXISTS training_records (
	model_id					SERIAL	PRIMARY KEY ,
    label_names  				varchar[] ,
	image_counts_for_labels		int[],
	final_accuracy				float,		
	final_validation_accuracy	float,
	final_loss					float,
	final_validation_loss		float,
	saved_model_path text,
	initial_requested_user_id int references users(user_id),
    purchased_user_ids    int[],
	creation_date 		timestamp,
	note				text
);
	
	
	
-- Altered training_records to include downloadable modelPath and downloadable result plot

ALTER TABLE training_records	
ADD COLUMN downloadable_model_link text,
ADD COLUMN downloadable_plot_link text
;

SELECT * FROM training_records;