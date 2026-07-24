name := "disflr"

version := "0.1.0"

organization := "edu.umich"

scalaVersion := "2.12.18"

val sparkVersion = "3.5.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql"  % sparkVersion % "provided",
  "org.scalanlp"     %% "breeze"     % "2.1.0"      % "provided"
)
