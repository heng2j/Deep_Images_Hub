-- Queries



-- Select Statemens
-- Select to check if the label exist
SELECT count(label_name)
FROM labels
WHERE label_name = 'Think_thin_high_protein_caramel_fudge';



-- update label's count 
						
UPDATE labels 
SET image_count = image_count + 10
WHERE label_name = 'Apple';

SELECT * 
FROM labels
WHERE label_name = 'Coconut'


-- Select parent Recursively 

WITH RECURSIVE labeltree AS ( 
	 SELECT parent_name
     FROM labels 
     WHERE label_name = 'Fox'
     UNION ALL
     SELECT l.parent_name 
     FROM labels l 
     INNER JOIN labeltree ltree 
	 ON ltree.parent_name = l.label_name 
     WHERE l.parent_name IS NOT NULL) 
                
	 SELECT * 
     FROM labeltree;



