# Zillow dataset information

## Data:

[https://www.zillow.com/research/data/](https://www.zillow.com/research/data/)

## MARKET HEAT INDEX

The market heat index is a time series dataset that aims to capture the balance of for-sale supply and demand in a given market. A higher number means the market is more tilted in favor of sellers. It relies on a combination of engagement and listing performance inputs to provide insights into current market dynamics. It is calculated for single-family and condo homes. For more information, see [Market Heat Index Methodology](https://www.zillow.com/research/market-heat-index-methodology-34057/)

    How to interpret the Heat Index data
    The metric assumes an index value typically ranging from 0-100. A higher value indicates the market is more tilted in favor of sellers. 

    We consider: 

    A score of 70 or above to be a “strong sellers market.” 
    A score from 55 to 69 to be a “sellers market.”
    A score from 44 to 55 to be a “neutral market.” 
    A score from 28 to 44 to be a “buyers market.”
    A score of 27 or below to be a “strong buyers market.” 
    While the range of these numbers is somewhat arbitrary, the relation of these indexes gives us information about how a given market compares to others, or to itself in the past, in terms of “temperature.” For example, we can make the following comparisons with the current metric:

    Region A has a higher temperature than region B, so region A has more buyer competition than B and market balance is therefore tilted more in favor of sellers.
    Region A is up 10% month over month, indicating the competition in the region is increasing. This suggests there are more potential buyers per available listing than the previous month.
    Region A is down 5% year over year, indicating that compared to last year it is slightly easier to buy a home. 

## DAYS ON MARKET AND PRICE CUTS

Days to Pending: How long it takes homes in a region to change to pending status on Zillow.com after first being shown as for sale. The reported figure indicates the number of days (mean or median) that it took for homes that went pending during the week being reported, to go pending. This differs from the old “Days on Zillow” metric in that it excludes the in-contract period before a home sells.

Days to Close (mean/median): Number of days between the listing going pending and the sale date.
Share of Listings With a Price Cut: The number of unique properties with a list price at the end of the month that’s less than the list price at the beginning of the month, divided by the number of unique properties with an active listing at some point during the month.

Price Cuts: The mean and median price cut for listings in a given region during a given time period, expressed as both dollars ($) and as a percentage (%) of list price.