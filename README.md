# Impact Evaluation: Detecting errors within PowerBI reports with computer vision models

This impact evaluation reviews a AI/ML project completed recently (October 2024). 
For publishing purposes, and data has been anonymized.

## Overview

### Positive Impact

Since the completion of this project, it has been successfully deployed for use across all our companies reports. 
This deployment has significantly enhanced our teams ability to detect visual errors within reports, and therefore benefits the business by providing error-free reports. 
Prior to this, we as a team only found out about report visual errors via two ways: naturally working with our reports and discovering a visual error or a visual error being reported by one of the report users.
However, with the vision model in place, we can now detect these issues alot easier.

Intruducing this model has improved the team's effiency too. As if the model was not in place, then a team member would have to manually go through the reports to discover these visual errors. The benefits have been amplified as of late too, as the number of reports being created by the team has risen sharply. This project has, therefore, led to significant time and resource savings.

### Negative Impact

While the project has achieved positive outcomes, there has been some questions on model accuracy, resulting in the potential of the model missing some visual errors. However I empahsize the word potential, as the probability of this occuring seems very low according to my evaluation on the model.
So in total the negative impacts of this project have been very little.

With some data/AI solutions scalability can be an issue but again as I have utilised databricks and can easily scale out the workload across more machines this will no be an issue. Also the retraining process is as simple as adding more data and re-running a notebook, which is simplier than most ai systems which use models.

So overall, the model just needs to be closely monitored during its early phase where it has just been deployed. After this phase, the model is not monitored as closely but if visual errors are found not be picked up by the model, extensive work should be done to understand why not and come up with an fix (i.e. adding more diverse training data to capture more unique conditions, regularize the model to make sure no overfitting is taking place, etc).

## Conclusion

In conclusion, this AI/ML project has had a significant positive impact on our company's reporting standards and the teams effiency, allowing for increased workloads and better handling of a large amount of business reports. Though, it has been noted that if errors are not picked up by the productionised model then extensive work should be put in place to discover why.

##  Bibliography
Google AppSheet | Build apps with no code (no date) AppSheet. Available at: https://about.appsheet.com/home/
ChatGPT (OpenAI) assisted in content restructuring. Available at: https://www.openai.com/research/chatgpt



# Original Documentation

# Contents

[Introduction](#introduction)

[Project Summary](#project-summary)

[Issue at Hand](#issue-at-hand)

[Solution](#solution)

[Outcome](#outcome)

[Pipeline Overview](#pipeline-overview)

[A High Level Look at the Pipeline](#a-high-level-look-at-the-pipeline)

[A Deeper Dive Into the Pipeline](#_Toc124888952)

[Desired Output Structure](#desired-output-structure)

[Ingesting Data from Google Sheets](#ingesting-data-from-google-sheets)

[Preparing the Branches Import](#preparing-the-branches-import)

[Applying Business Logic](#applying-business-logic)

[Bringing it all Together](#bringing-it-all-together)

[Validating the Outputs](#validating-the-outputs)

[Displaying Results to the User](#displaying-results-to-the-user)

[Conclusion](#conclusion)

[Outline](#outline)

[Considerations](#considerations)

[Bibliography](#bibliography)


# Project Background

A big part of a data team within a business is to provide insighful reports company wide.

These reports cover a wide range of topics/data and departments/teams use them for insights, performance evaluation and creating new strategies. These reports we offer to the business are considered as products which we deliver and maintain.

The process for building reports is quite lengthy as we try to follow best standards and clear processes when it comes to creating a new dashboard. To quickly outline the process it usually starts with a get together to understand report requirements with the stakeholders, gathering the data, feeding the data through our custom built data pipelines, dimesnionally modelling (Kimball methodology), creating report measures in DAX and then finally creating the PowerBI report. This project focuses on the latter stage, but probably one of the most important stages too. If there is an visual error on the report, report users are likely to abandon the report very quickly and turn back to other solutions like Excel. Meaning all the hard work which was done previously to get the data into that report, has basically been made pointless. Therefore it is important we are able to identify these visual report errors within PowerBI quickly and get them solved.

For reference the PowerBI visual error symbol looks like a cross and is what we will be focussing on to identify:

![PowerBI visual error reference](screenshots/powerbi-visual-error.png)

To detect these visual errors it could be taken away as a manual task to do everytime we release an update to our reports, however like I have already mentioned this would make the team more ineffiecent. Also with the increasing number of reports which we have created and now maintain, this manual task would get more inefficient as time went. 

Therefore the automatic system we wanted to put in place was a neural network, like a convulitonal or recurrent neural network (CNN/RNN respectively), which could identify objects. Then with this model I would use transfer learning to modify a specific set of weights to make the model more specific to our use case and data. This model should then be able to identify objects like PowerBI visual errors, graphs, tables and so on. The productionisation of this model would also be a big factor, as the model would need to be easily accessible so that inference (prediction) would be easy. There are several possible ways to do this, but as our current data solution already uses unity catalog within databricks to manage the data at its different stages in the pipeline, I felt it would be best to store the model in unity catalog. This will be discussed more later on.

Reports being one of the most important products we offer as the data team, using AI/ML to ensure their consistency and standards are maintained, was a good use of machine learning.

# Project Summary

## Issue at Hand

Updating capacities using the newer logic was heavily reliant on one
team member, meaning that the frequency in which they were updated
wasn't as high as it could be. The strain on this team member will only
increase as the process is rolled out to further regions.

## Solution

The process should be moved from the Excel tool to a cloud based
solution, allowing for anyone within the team to update the input
parameters, then request an ingestion of the output into the application
database.

## Outcome

A pipeline was created that efficiently implements the proposed
solution, for the primary region where the company trades. Data has been
validated against the existing tool to be accurate.

# Pipeline Overview

## A High Level Look at the Pipeline

A lifecycle approach will be taken to build the pipeline. This starts by
getting the data from the sources and ingesting it, from there it will
be transformed before going onto the final steps where the results are
used to inform business decisions.

![Graphical user interface, diagram, application Description
automatically generated](media/image1.png)

(*2. The Data Engineering Lifecycle*, no date)

The existing tooling has user inputs inside a spreadsheet for

-   Total physical space available per location

-   An array of parameters for tweaking the outputs

The physical space available is then combined with transactional data to
generate the capacity outputs.

The transactional data is imported into the existing tool via an OLAP
cube, which contains precalculated data designed to allow for historical
analysis (*The Rise and Fall of the OLAP Cube*, 2020). The underlying
tables are stored in a star schema, where the data is denormalized and
stored according to the data type.

The star schema approach classifies the tables within a model as either
a fact or a dimension (Sanchez-Ayala, 2020).

Dimension tables contain categorical data (e.g. products, locations, or
dates), there could be multiple dimension tables linked to a single fact
table.

Fact tables contain quantitative data relating to business processes
(e.g. quantity and value of units sold, or on hand), a fact table will
typically contain foreign keys that relate to the dimensional tables.

A basic overview of the pipeline would be.

![Basic ETL diagram. ](media/image2.png)

To keep permissions simple the pipeline will be contained within GCP, a
cloud computing platform (*Cloud Computing Services*, no date), and
Google Workspace which is a set of productivity tools designed to
enhance collaboration offered by Google (*Announcing Google Workspace,
everything you need to get it done, in one location*, no date).

The transactional data required is held within BigQuery, a scalable data
warehouse solution provided by Google (*What is BigQuery?*, no date).
BigQuery has a feature called "External Data Sources" which will allow
us to join our fairly static transactional data with the variable user
inputs in the Google Sheet without having to create a GUI for this
process (*Introduction to external data sources \| BigQuery \| Google
Cloud*, no date).

## A Deeper Dive Into the Pipeline

### Desired Output Structure

The final output is heavily based on the outputs of the existing
tooling, some extra columns will be added as foreign keys for ease of
importing the data back into the application database.

| Output Column              | Source Type      | Source Table(s)                                                   | Noteable Column(s)                    |
|----------------------------|------------------|-------------------------------------------------------------------|---------------------------------------|
| branchkey                  | Application Data | branches                                                          | branchkey                             |
| branchid                   | Application Data | branches                                                          | branchid                              |
| branch                     | Application Data | branches                                                          | branch_name                           |
| cabinet_meterage           | User Input       | gsheet_branches                                                   | cabinet_meterage                      |
| shop_floor_shelf_count     | User Input       | gsheet_branches                                                   | shop_floor_shelf_count                |
| treat as new               | User Input       | gsheet_branches                                                   | treat as new                          |
| cabinet_multiplier         | Calculated       | gsheet_branches                                                   | cabinet_meterage                      |
| media_multiplier           | Calculated       | gsheet_branches                                                   | shop_floor shelf_count                |
| category_id                | Application Data | category                                                          | category_id                           |
| box_category               | Application Data | category                                                          | box_category                          |
| category_location          | User Input       | gsheet_category_types                                             | category_location                     |
| total stock                | Calculated       | aio_fact                                                          | quantity                              |
| average_stock_per_location | Calculated       | aio_fact, gsheet_branches                                         | quantity                              |
| minimum_capacity           | Calculated       | aio_fact, gsheet_branches, gsheet_controls                        | quantity, minimum_capacity_percentage |
| maximum_capacity           | Calculated       | aio_fact, gsheet_branches, gsheet_controls                        | quantity, maximum_capacity_percentage |
| capacity_ratio             | Calculated       | aio_fact, gsheet_branches, gsheet_controls, gsheet_category_types |                                       |
| raw cap                    | Calculated       | aio_fact, gsheet_branches, gsheet_controls, gsheet_category_types |                                       |
| new_capacity               | Calculated       | aio_fact, gsheet_branches, gsheet_controls, gsheet_category_types |                                       |

### Ingesting Data from Google Sheets

Google Sheets will be used as a storage location for user inputs, this
was picked as spreadsheet like software is ubiquitous in the business
world.

The variables that are being ingested are across three tabs within a
workbook container, once these have been brought into BigQuery, we can
combine them with application generated data to apply business logic.

The initial tab contains columns specifying attribute names in both a
user and machine-readable format, assigned values, and finally a data
type.

| friendly_name       | attribute                   | value | attribute_type |
|---------------------|-----------------------------|-------|----------------|
| Size Weighting      | size_weighting_percentage   |       | percentage     |
| Sales Weighting     | sales_weighting_percentage  |       | percentage     |
| x Shelf Threshold   | x_shelf_threshold_number    |       | number         |
| x Percentage Normal | x_target_percentage         |       | percentage     |
| Cap Buffer          | capacity_buffer_percentage  |       | percentage     |
| Min cap             | minimum_capacity_percentage |       | percentage     |
| Max cap             | maximum_capacity_percentage |       | percentage     |

The second tab contains a list of branch names, totals for the cabinet
meterage and shelving counts, and a flag for if sales data should be
considered when calculating the output for that branch.

| branch | cabinet_meterage | shelf_count | treat_as_new |
|--------|------------------|-------------|--------------|
| A      | 11.04            | 203         | FALSE        |
| B      | 9.08             | 251         | FALSE        |
| C      | 6.32             | 245         | TRUE         |
| D      | 9.08             | 264         | FALSE        |
| E      | 11.84            | 630         | FALSE        |

The third tab contains product categories, additional grouping
information for filtering, and columns for dictating business logic. Of
the two business logic columns only category location is used within
this pipeline, to act as a predicate when applying the logic.

| category    | category_grouping | supercat   | spine_type | category_location |
|-------------|-------------------|------------|------------|-------------------|
| Category 1  |                   | Supercat 1 |            | cabinet           |
| Category 2  |                   | Supercat 1 |            | cabinet           |
| Category 3  |                   | Supercat 2 |            | cabinet           |
| Category 4  |                   | Supercat 3 |            | cabinet           |
| Category 5  |                   | Supercat 2 |            | cabinet           |
| Category 6  |                   | Supercat 1 |            | cabinet           |
| Category 7  |                   | Supercat 1 | 1          | floor             |
| Category 8  |                   | Supercat 4 | 1          | floor             |
| Category 9  |                   | Supercat 1 | 1          | floor             |
| Category 10 |                   | Supercat 4 | 2          | floor             |
| Category 11 |                   | Supercat 5 |            | NA                |

To bring these into BigQuery, they will be linked as external data
sources. The data type for all columns will be set as string, this is
due to the mixed types on the initial tab, and user input data with
limited validation rules.

These are brought into BigQuery using DDL create the external table
references.

```sql
CREATE EXTERNAL TABLE `autocaps.gsheet_controls`
(
    friendly_name STRING,
    attribute STRING,
    value STRING,
    attribute_type STRING
    )
OPTIONS(
    sheet_range="controls",
    skip_leading_rows=1,
    format="GOOGLE_SHEETS",
    uris=["https://docs.google.com/spreadsheets/d/[sheet_key]/"]  
    )
```

Once the links have been established for all tabs within the workbook,
the data can be queried just like a native table.

#### Transforming the Controls

A view is created over the top of the controls table to transform the
values for use in later stages.

Rows specified as having an attribute type of percentage is reduced down
to a decimal form. Dates are parsed from DD/MM/YYYY to standardised
format of YYYY-MM-DD (*ISO - ISO 8601 --- Date and time format*, no
date).

```sql
with
    input_parameters as (
        select attribute,
        case lower(attribute_type)
            when "percentage"
                then cast(cast(value as numeric)/100 as string)
            when "date"
                then cast(parse_date('%d/%m/%Y',value) as string)
            else value
        end as value
        )

select * from input_parameters
```

### Preparing the Branches Import

As with the controls table, a view is created over the top of the
branches data that is ingested from Google sheets.

The first step in this view is to construct a CTE that includes a branch
key, and location id for the branches. This data comes from the managed
data warehouse that contains application data.

Some light cleaning needs to be done on the branch column, as some
branches have double spaces.

```sql
with branches as (
    select branches.branchid,
           branches.branchkey,
           -- Some store names have double spaces, the replace is to trim those to havinbg one space. 
    replace(branches.branch_name,"  "," ") as branch,
    lblo.locationid
    from `branches`
    inner join `lblo` using(branchid))

select * from branches
```


The next step in this view is to clean and deduplicate the data from the
google sheet. Double spaces are removed from the branch column, data
types are established for the columns.

To clean this further, there is a predicate that specifies only rows
containing a value in the branch column will be returned. Then to
deduplicate the data we're using the group by clause to aggregate the
data, then excluding any rows using the having filter to return only
rows where the occurrence is equal to 1 (Gupta, 2019).

```sql
with google_sheet as 
(
    select replace(branch,"  ", " ") as branch,
           sum(cast(cabinet_meterage as numeric)) as cabinet_meterage,
           sum(cast(shelf_count as numeric)) as shop_floor_shelf_count,
           cast(max(treat_as_new) as bool) as treat_as_new
    from `autocaps.gsheet_branches`
    where branch is not null
    group by branch
    having count(branch) = 1
)
select * from google_sheet
```

| branch | cabinet_meterage | shelf_count | treat_as_new |
|--------|------------------|-------------|--------------|
| A      | 11.04            | 203         | FALSE        |
| B      | 9.08             | 251         | FALSE        |
| C      | 6.32             | 245         | TRUE         |
| D      | 9.08             | 264         | FALSE        |
| E      | 11.84            | 630         | FALSE        |

We then bring the category location field from the category types table.

```sql
with category_types as (
    select category,
           category_location
    from `autocaps.gsheet_category_types`
)
select * from category_types
```
| category | category_location |
|----------|-------------------|
| 1        | cabinet           |
| 2        | cabinet           |
| 101      | floor             |

These three CTEs are joined together to provide a reference that we can
start applying some business logic to.

```sql
with filtered_branches as (
    select branches.branchkey,
           branches.locationid,
           branches.branch,
           google_sheet.cabinet_meterage,
           google_sheet.treat_as_new
    from branches
        -- We're doing an inner join to exclude any stores not filtered by previous steps.
        -- Bigquery is case sensitive, so the join is done using the lower() function
    inner join google_sheet on lower(branches.branch) = lower(google_sheet.branch)
)
select * from filtered_branches
```
| branchkey | branchid | locationid | branch | cabinet_meterage | shop_floor_shelf_count | treat_as_new |
|-----------|----------|------------|--------|------------------|------------------------|--------------|
| AA        | 1        | 1001       | A      | 10.46            | 256                    | FALSE        |
| BB        | 2        | 1003       | B      | 10.46            | 405                    | FALSE        |
| CC        | 3        | 1005       | C      | 6.32             | 195                    | TRUE         |
| DD        | 4        | 1007       | D      | 8.5              | 305                    | FALSE        |


The final select of this view returns an output that joins three index
keys in front of the branch name, alongside the imported variables from
the Google sheet, and finally creates two calculated columns using
nested sub queries of a prior CTE.

```sql
select branchkey,
       branchid,
       locationid,
       branch,
       cabinet_meterage,
       shop_floor_shelf_count,
       treat_as_new,
       safe_divide(cabinet_meterage,(select safe_divide(sum(cabinet_meterage),count(*)) from filtered_branches)) as cabinet_multiplier,
       safe_divide(shop_floor_shelf_count,(select safe_divide(sum(shop_floor_shelf_count),count(*)) from filtered_branches)) as media_multiplier
       from filtered_branches
```

| branchkey | branchid | locationid | branch | cabinet_meterage | shop_floor_shelf_count | treat_as_new | cabinet_multiplier | media_multiplier |
|-----------|----------|------------|--------|------------------|------------------------|--------------|--------------------|------------------|
| AA        | 1        | 1001       | A      | 10.46            | 256                    | FALSE        | 0.9                | 0.85             |
| BB        | 2        | 1003       | B      | 10.46            | 405                    | FALSE        | 1.29               | 0.960            |
| CC        | 3        | 1005       | C      | 6.32             | 195                    | TRUE         | 0.95               | 1.250            |
| DD        | 4        | 1007       | D      | 8.5              | 305                    | FALSE        | 0.79               | 1.410            |

### Applying Business Logic

The business logic is applied through two further views, the first view
aggregates the total stock figures on a category level, then performs
calculations using the user inputs.

The user inputs are pivoted from a row to a columnar structure.
```sql
    parameters_pivot as (
        select cast(minimum_capacity_percentage as numeric) as minimum_capacity_percentage,
               cast(maximum_capacity_percentage as numeric) as maximum_capacity_percentage,
               cast(capacity_buffer_percentage as numeric) as capacity_buffer_percentage
        from (select attribute,value from `autocaps.stg_controls`)
        pivot(max(value) for attribute in ('minimum_capacity_percentage','maximum_capacity_percentage','capacity_buffer_percentage'))
)
```
| minimum_capacity_percentage | maximum_capacity_percentage | capacity_buffer_percentage |
|-----------------------------|-----------------------------|----------------------------|
| 0.15                        | 2.6                         | 0.1                        |



The stock quantities are then aggregated and a cross join is performed.

```sql
    aio_stock as (
        select  category_id,
                box_category,
                category_location,
                supercat,
                sum(stock_quantity) as total_stock
        from `autocaps.stg_aio_data`
        group by category_id, box_category, category_location, supercat),
    stock as (
        select category_id,
               box_category,
               supercat,
               category_location,
               total_stock,
               minimum_capacity_percentage,
               maximum_capacity_percentage,
               capacity_buffer_percentage
        from aio_stock
        cross join parameters_pivot
)
```

A count of the branches is also
imported.
```sql
branches as (select count(*) as branch_count from `autocaps.stg_branches_cabinets`)
```

The final select of this view uses the stock information, user inputs
and count of branches to perform calculations, which can then be used in
subsequent steps.

A table of ratios is calculated also, using similar steps where the user
inputs are imported, pivoted, and cross joined against transactional
data.

### Bringing it all Together

The final stage of the pipeline is a view that ties all of the prior
steps together, it utilises a similar structure to the previous queries
by utilising a selection of CTEs, joins and a final application of
business logic.

The view can then be used in user facing tools, where the user needs
dynamic feedback after the inputs are changed.

A scheduled task is also created to take a snapshot at midnight daily
and store it in a table, this allows for lower overheads when displaying
the data to users that don't need to change the inputs. The table also
provides a stable source for reverse ETL pipelines, which is a process
where business logic is applied to data, then the results are fed back
into the application databases (*9. Serving Data for Analytics, Machine
Learning, and Reverse ETL*, no date).

### Validating the Outputs

As this is a translation of an existing Excel tool, the data can be
validated against that.

The data is first imported to Excel using the get data from other
workbook feature of Power Query

| Branch | Category 1 | Category 2 | Category 3 |
|--------|------------|------------|------------|
| A      | 2          | 5          | 59         |
| B      | 2          | 4          | 45         |
| C      | 2          | 3          | 37         |
| D      | 2          | 5          | 42         |
| E      | 2          | 2          | 37         |

The data is in a pivot format, to be able to compare this to the results
being retrieved from BigQuery we need to perform some transformations.

Firstly we remove the double spaces from the branch names, then due to
the import not being a table in the source workbook there is extra rows
that need to be filtered out, finally the data is unpivoted.

| Branch | Attribute  | Value |
|--------|------------|-------|
| A      | Category 1 | 2     |
| A      | Category 2 | 5     |
| A      | Category 3 | 59    |
| B      | Category 1 | 2     |
| B      | Category 2 | 4     |
| B      | Category 3 | 45    |
| C      | Category 1 | 2     |
| C      | Category 2 | 3     |
| C      | Category 3 | 37    |
| D      | Category 1 | 2     |
| D      | Category 2 | 5     |
| D      | Category 3 | 42    |
| E      | Category 1 | 2     |
| E      | Category 2 | 2     |
| E      | Category 3 | 37    |

Excel doesn't support BigQuery natively, so an ODBC driver is required
to help harness the power of cloud computing within Excel, this is done
using the Simba drivers (*ODBC and JDBC drivers for BigQuery*, no date).
Once these are installed and configured, the data is imported to Excel
via an ODBC SQL query.

The two sources are then joined together using the branch and category
columns, a custom column is added to check for equality.

This is done twice, once with the Excel import as the left table, and
once with the BigQuery import as the left table. By doing that it can be
asserted that there is no extra branches or categories in either source.

As the preview in Power Query only displays 1000 rows, these need to be
loaded into Excel to validate the full data set of approximately 100,000
rows.

### Displaying Results to the User

Looker Studio is being used to display the outputs of the query to the
user to allow them to verify the impact of the chosen inputs.

# Conclusion

## Outline

The creation of this pipeline was an exercise in reducing the toil of a
task performed through an Excel tool. The tool is rather complex
requiring the user to have a powerful local machine, and be aware of the
nuances for importing the transactional data.

This means that the task falls upon one member within the department,
creating a bottleneck when the task needs to be completed. To open up
this task to be performed by other members within the team a cloud first
approach was taken.

Having an existing tool helped in establishing the structure of the
output required, along with the data sources needed.

Validation was done throughout the steps of the pipeline, some issues
were identified and remedied. Most notably the conversion of data types
from a floating point value where the calculations were being performed
on an approximation and fractional basis to a numeric datatype for
increased precision (*Data types \| BigQuery*, no date).

Looker Studio was chosen to display the outputs to the user, this
decision was based on the ease of permissions management, deep
integration with the Google ecosystem, and familiar user interface (*The
next evolution of Looker, your unified business intelligence platform*,
no date).

While the dynamic view at the end of the pipeline could have been used
as a source for reverse ETL, it was decided that a daily snapshot table
would be more reliable.

## Considerations

The pipeline is rather complex, to the point where limits were being hit
for sub queries.

Some refactoring resolved this, but the trade-off was the loss of
flexibility. For example there is a few steps where views could be
replaced with table-valued functions which behave similarly, but allow
for parameters to be passed as a variable to use (*Table functions \|
BigQuery \| Google Cloud*, no date).

A solution to reduce some complexity of the dynamic queries would be to
identify which stages of the pipeline could be converted to snapshots.
Importing and aggregating the transactional data on a weekly basis would
vastly reduce the resources used.

The SQL was written in the BigQuery web interface, which is great for
ad-hoc queries, but isn't entirely suited for writing pipelines. Using a
tool that is suited for this process would be more effective for
managing the version history of the queries, along with easy generation
of data lineage diagrams.

My choice for this would be dbt (data build tool), a command line tool
designed to enable transformation of data more efficiently (*What,
exactly, is dbt?*, no date).

#  Bibliography

*2. The Data Engineering Lifecycle* (no date). Available at:
https://learning.oreilly.com/library/view/fundamentals-of-data/9781098108298/ch02.html
(Accessed: 9 January 2023).

*9. Serving Data for Analytics, Machine Learning, and Reverse ETL* (no
date). Available at:
https://learning.oreilly.com/library/view/fundamentals-of-data/9781098108298/ch09.html
(Accessed: 17 January 2023).

*Announcing Google Workspace, everything you need to get it done, in one
location* (no date) *Google Workspace Blog*. Available at:
https://workspace.google.com/blog/product-announcements/introducing-google-workspace
(Accessed: 18 November 2022).

*Cloud Computing Services* (no date) *Google Cloud*. Available at:
https://cloud.google.com/ (Accessed: 18 November 2022).

*Data types \| BigQuery* (no date) *Google Cloud*. Available at:
https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types
(Accessed: 14 January 2023).

Garza, N. (2020) *Planograms 101: What They Are and Why They're
Important for Your Brand*, *ePac Flexible Packaging*. Available at:
https://epacflexibles.com/planograms-101-what-they-are-and-why-theyre-important-for-your-brand/
(Accessed: 21 November 2022).

Gupta, R. (2019) 'Different ways to SQL delete duplicate rows from a SQL
Table', *SQL Shack - articles about database auditing, server
performance, data recovery, and more*, 30 August. Available at:
https://www.sqlshack.com/different-ways-to-sql-delete-duplicate-rows-from-a-sql-table/
(Accessed: 9 January 2023).

*Introduction to external data sources \| BigQuery \| Google Cloud* (no
date). Available at:
https://cloud.google.com/bigquery/docs/external-data-sources (Accessed:
21 November 2022).

*ISO - ISO 8601 --- Date and time format* (no date) *ISO*. Available at:
https://www.iso.org/iso-8601-date-and-time-format.html (Accessed: 14
January 2023).

*ODBC and JDBC drivers for BigQuery* (no date) *Google Cloud*. Available
at: https://cloud.google.com/bigquery/docs/reference/odbc-jdbc-drivers
(Accessed: 12 January 2023).

Sanchez-Ayala, M. (2020) 'Data Modeling: The Star Schema', *Medium*, 8
April. Available at:
https://medium.com/@marcosanchezayala/data-modeling-the-star-schema-c37e7652e206
(Accessed: 14 January 2023).

*Table functions \| BigQuery \| Google Cloud* (no date). Available at:
https://cloud.google.com/bigquery/docs/reference/standard-sql/table-functions
(Accessed: 14 January 2023).

*The next evolution of Looker, your unified business intelligence
platform* (no date) *Google Cloud Blog*. Available at:
https://cloud.google.com/blog/products/data-analytics/looker-next-evolution-business-intelligence-data-studio
(Accessed: 18 November 2022).

*The Rise and Fall of the OLAP Cube* (2020) *The Holistics Blog*.
Available at:
https://www.holistics.io/blog/the-rise-and-fall-of-the-olap-cube/
(Accessed: 14 January 2023).

*What, exactly, is dbt?* (no date) *Transform data in your warehouse*.
Available at: https://www.getdbt.com/blog/what-exactly-is-dbt/
(Accessed: 16 January 2023).

*What is BigQuery?* (no date) *Google Cloud*. Available at:
https://cloud.google.com/bigquery/docs/introduction (Accessed: 18
November 2022).
