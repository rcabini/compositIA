<?php
	ini_set('display_errors', 1);
	ini_set('display_startup_errors', 1);
	error_reporting(E_ALL);
	set_time_limit(600);
	$PROC_PATH = "/var/www/html/compositia/proc/";
	$UPLOAD_PATH = "/var/www/html/compositia/upload/";

	$fn = $_GET["fn"];
	echo $fn.'<br/>';
	$res1 = '';
	$retval = '';
	echo 'bash '.$PROC_PATH.'runscores2.sh '.$UPLOAD_PATH.$fn.'<br/>';
	exec('bash '.$PROC_PATH.'runscores2.sh '.$UPLOAD_PATH.$fn, $res1, $retval);
	print_r($res1);
	echo $res1[0].'<br/>';
	echo $retval.'<br/>';
	echo 'fine';
	
?>