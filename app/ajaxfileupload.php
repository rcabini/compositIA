<?php

if(isset($_FILES['file']['name'])){
   // file name
   $filename = "1".time()."_".$_FILES['file']['name'];

   // Location
   $location = 'upload/'.$filename;

   // file extension
   $file_extension = pathinfo($location, PATHINFO_EXTENSION);
   $file_extension = strtolower($file_extension);

   // Valid extensions
   $valid_ext = array("png", "gz", "nii","nii.gz");

   $response = 0;
   if(in_array($file_extension,$valid_ext)){
      // Upload file
      if(move_uploaded_file($_FILES['file']['tmp_name'],$location)){
         $response = $filename;
      } 
   }

   echo $response;
   exit;
}