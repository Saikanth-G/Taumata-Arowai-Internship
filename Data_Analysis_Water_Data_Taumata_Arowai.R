# Install the following packages
install.packages("readxl")
install.packages("dplyr")
install.packages("ggplot2")

# Load the following libraries
library(readxl) # to import data from excel files
library(dplyr) # for data manipulation
library(ggplot2) # for building statistical graphs

--------------------------------------------------------------------------

# Read the excel file from the source
file_path <- "C:/Users/91966/Documents/SKS/Internship Docs, Education New Zealand/cleaned_combined_water_data.xlsx"
data <- read_excel(file_path)

## Understanding the data
View(data) # opens the data in another tab
head(data) # provides the first row of the data, to gauge and understand
str(data) # to display the internal structure of the R object

# Now that the data is loaded, we find a summary of dataset
summary(data) # for quick exploratory data analysis of the dataset

# Show column names of the dataset
colnames(data)

# Check for missing values in the dataset
colSums(is.na(data))

# what is the dimension of the dataset
dim(data)
# so the dataset has 30,779 rows and 30 columns

--------------------------------------------------------------------

##SHOWING THE GEOGRAPHICAL SPREAD OF AVAILABLE DATA
library(ggplot2)

ggplot(data, aes(x = Longitude, y = Latitude)) +
  geom_point(alpha = 0.5, color = "steelblue", size = 0.8) +
  coord_fixed() +
  labs(
    title = "Environmental Monitoring Sites",
    x = "Longitude",
    y = "Latitude"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  )

--------------------------------------------------------------------
##HOW CLOSE ARE THE EXTERNAL DATASETS TO TO THE INTERNAL WATER SUPPLIES

ggplot(data, aes(x = Match_Distance_km)) +
  geom_histogram(bins = 30, fill = "navy", color = "white") +
  labs(title = "Distribution of Spatial Match Distances",
       x = "Match Distance (km)", y = "Number of Observations")

-----------------------------------------------------------------------
## Time series of indicator values for ASPM indicator

library(dplyr)

# Filter one common indicator
indicator_subset <- data %>% filter(Indicator_Name == "ASPM", !is.na(Indicator_Value))

ggplot(indicator_subset, aes(x = Sample_Date_Time, y = Indicator_Value)) +
  geom_line(color = "darkorange") +
  labs(title = "ASPM Indicator Value Over Time",
       x = "Date", y = "ASPM Value")

---------------------------------------------------------------------------------------------------

## Time series of indicator values for MCI indicator
# Filter one common indicator
indicator_subset_1 <- data %>% filter(Indicator_Name == "MCI", !is.na(Indicator_Value))

ggplot(indicator_subset, aes(x = Sample_Date_Time, y = Indicator_Value)) +
  geom_line(color = "darkblue") +
  labs(title = "MCI Indicator Value Over Time",
       x = "Date", y = "MCI Value")

----------------------------------------------------------------------------------------------------

## Time series of indicator values for the E COLI indicator
# Filter both "E.coli" and "E. coli" (handles both spellings)
ecoli_data <- data %>%
  filter(Indicator_Name %in% c("E.coli", "E. coli"),
         !is.na(Indicator_Value),
         !is.na(Sample_Date_Time))

# Plot the time series
ggplot(ecoli_data, aes(x = Sample_Date_Time, y = Indicator_Value)) +
  geom_line(alpha = 0.3, color = "blue") +
  geom_smooth(method = "loess", se = FALSE, color = "green", linewidth = 1) +
  labs(
    title = "Time Series of E. coli Indicator Values (2004–2023)",
    x = "Date",
    y = "E. coli Value"
  ) +
  theme_minimal()

-----------------------------------------------------------------------------------------
## Bar Chart of Most Frequently Reported Indicators
data %>%
  count(Indicator_Name, sort = TRUE) %>%
  slice_max(n, n = 10) %>%  # Top 10 by count
  ggplot(aes(x = reorder(Indicator_Name, n), y = n)) +
  geom_bar(stat = "identity", fill = "#2a9d8f", width = 0.7) +
  coord_flip() +
  labs(
    title = "Top 10 Reported Environmental Indicators",
    x = "Indicator Name",
    y = "Number of Observations"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    axis.title.x = element_text(margin = margin(t = 10)),
    axis.title.y = element_text(margin = margin(r = 10)),
    panel.grid.major.y = element_blank()
  )

## Bar chart of the 10 least reported indicators
data %>%
  count(Indicator_Name, sort = TRUE) %>%
  slice_min(n, n = 10) %>%
  ggplot(aes(x = reorder(Indicator_Name, n), y = n)) +
  geom_bar(stat = "identity", fill = "#e76f51", width = 0.7) +
  coord_flip() +
  labs(
    title = "10 Least Reported Environmental Indicators",
    x = "Indicator Name",
    y = "Number of Observations"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    axis.title.x = element_text(margin = margin(t = 10)),
    axis.title.y = element_text(margin = margin(r = 10)),
    panel.grid.major.y = element_blank()
  )
---------------------------------------------------------------------------
## Summary stats for ASPM, MCI, and E. coli
# Standardize spelling for E. coli
data_cleaned <- data %>%
  mutate(Indicator_Name = ifelse(Indicator_Name %in% c("E.coli", "E. coli"), "E. coli", Indicator_Name))

# Filter for key indicators and drop missing values
filtered_data <- data_cleaned %>%
  filter(Indicator_Name %in% c("ASPM", "MCI", "E. coli"), !is.na(Indicator_Value))

# Get summary statistics by indicator
filtered_data %>%
  group_by(Indicator_Name) %>%
  summarise(
    Count = n(),
    Min = min(Indicator_Value, na.rm = TRUE),
    Q1 = quantile(Indicator_Value, 0.25),
    Median = median(Indicator_Value),
    Mean = mean(Indicator_Value),
    Q3 = quantile(Indicator_Value, 0.75),
    Max = max(Indicator_Value)
  )


