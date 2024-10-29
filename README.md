# Impact Evaluation: Data Engineering Project (2023)

This impact evaluation reviews a data engineering project completed earlier in 2023. 
For publishing purposes, screenshots have been converted to code blocks, and data has been anonymized.

## Overview

### Positive Impact

Since the completion of this project, it has been successfully deployed for use in four different countries. 
This deployment has significantly enhanced our company's agility when it comes to updating store capacities. 
Prior to this, capacity updates were limited to quarterly intervals for each region. 
However, with the new system in place, we can now perform these updates at more frequent intervals.

One notable improvement is the removal of a previously time-consuming manual task, that required a member of the management team to spend approximately half an hour to an hour per update. Even if we conservatively estimate just one update per month for each of the four regions over eight months, the time saved becomes substantial. In total, this translates to 32 instances where the task would have been performed, not accounting for the increased update frequency. The project has, therefore, led to significant time and resource savings.

### Negative Impact

While the project has achieved positive outcomes, there has been some challenges with scope creep, resulting in a slower-than-anticipated implementation.
As a result, full automation hasn't been achieved yet. 
Currently, there is a still a manual blocker for capacity updates, although once initiated the task runs in the background.

Additionally, the user interface of a spreadsheet has some limitations in terms of scalability.
There is a need for further work to streamline the UI and reduce friction.
One potential solution being explored is Google AppSheet, which offers a low-code approach and seamless integration with Google Cloud.
This approach aims to ensure that further technical debt is avoided.(Google AppSheet | Build apps with no code, no date)

## Conclusion

In conclusion, this data engineering project has had a significant positive impact on our company's capacity management processes, allowing for more frequent and efficient updates. However, it has also highlighted the need for continued improvements in automation and user interface design to fully leverage the project's potential benefits.

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

A major part of a retail business model is having the correct stock in the right locations

These stock movements however need to be constrained due to limitations
of physical space available in a location. We call these capacities,
historically this process was done by using domain knowledge.

The process for this would involve them drawing out a high-level
planogram, which is a diagram of the store layout in a manner to best
maximise sales (Garza, 2020). Once this layout has been drawn up, the
shelving allocated to each of the categories would be noted in a
spreadsheet tool which then provided an output of the capacities for
each category.

As the company has grown, it is no longer feasible to maintain this
process. To streamline the workflow a new method of calculating these
capacities was prototyped in an Excel spreadsheet, the logic in the
spreadsheet has been proven to be sound, but there is a bottleneck in
this process in that the spreadsheet is resource heavy, and can only be
used by one member of the team.

Being a foundational part of the business model, this appears to be a
good project to convert from a spread mart tool to an ETL pipeline.

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
