## 01_select_where

```sql
SELECT title, price_gbp, rating FROM books WHERE rating >= 4 AND price_gbp < 30 ORDER BY price_gbp;
```

```text
                           title  price_gbp  rating
             The Darkest Corners      11.33       5
          New Moon (Twilight #2)      12.86       5
                 Starving Hearts      13.99       5
                    The Epidemic      14.44       4
               Obsidian (Lux #1)      14.86       4
               Kill the Boy Band      15.52       5
        The Coming Woman A Novel      17.93       4
                 Maude 1883-1993      18.02       4
                   Boy Meets Boy      21.12       4
America's Cradle of Quarterbacks      22.50       4
                 The Requiem Red      22.65       4
                    The Silkworm      23.05       5
             The Emerald Mystery      23.15       4
        Harry Potter and the ...      24.17       5
    The Mysterious Affair at ...      24.80       4
                    Extreme Prey      25.40       5
   Cinder (The Lunar Chronicles)      26.09       5
            This Is Where It ...      27.12       4
                      The Secret      27.37       5
         A Fierce and Subtle ...      28.13       5
               South of Sunshine      28.93       4
          The Land of 10,000 ...      29.64       4
  Frostbite (Vampire Academy #2)      29.99       4
```

## 02_order_by_limit

```sql
SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 10;
```

```text
                           title  price_inr
Aristotle and Dante Discover ...    6133.77
                  A Piece of Sky    5988.18
                    Future Shock    5871.08
                Don't Get Caught    5839.42
           The Girl on the Train    5804.61
                 No Love Allowed    5765.58
                  The Alien Club    5739.20
             Girl Online On Tour    5641.08
  Aladdin and His Wonderful Lamp    5605.22
                  Lady Renegades    5595.72
```

## 03_distinct

```sql
SELECT DISTINCT rating FROM books ORDER BY rating;
```

```text
 rating
      3
      4
      5
```

## 04_between_in

```sql
SELECT title, price_gbp, category_id FROM books WHERE price_gbp BETWEEN 15 AND 30 AND category_id IN (1,2,3) ORDER BY price_gbp;
```

```text
                           title  price_gbp  category_id
               Kill the Boy Band      15.52            3
                     Set Me Free      17.46            3
        The Coming Woman A Novel      17.93            1
                 Maude 1883-1993      18.02            1
            The Cuckoo's Calling      19.21            2
       The Inefficiency Assassin      20.59            1
                   Boy Meets Boy      21.12            3
America's Cradle of Quarterbacks      22.50            1
            The Boys in the Boat      22.60            1
                 The Requiem Red      22.65            3
                    The Silkworm      23.05            2
             The Emerald Mystery      23.15            1
            Lola and the Boy ...      23.63            3
        Harry Potter and the ...      24.17            3
    The Mysterious Affair at ...      24.80            2
                    Extreme Prey      25.40            2
   Cinder (The Lunar Chronicles)      26.09            3
            This Is Where It ...      27.12            3
                      The Secret      27.37            1
                      The Haters      27.89            3
         A Fierce and Subtle ...      28.13            3
                         Burning      28.81            3
               South of Sunshine      28.93            3
          The Land of 10,000 ...      29.64            3
  Frostbite (Vampire Academy #2)      29.99            3
```

## 05_join

```sql
SELECT b.title, c.category_name, b.rating, b.price_inr FROM books b JOIN categories c ON b.category_id=c.category_id ORDER BY b.rating DESC, b.price_inr DESC LIMIT 10;
```

```text
                           title category_name  rating  price_inr
Aristotle and Dante Discover ...   Young Adult       5    6133.77
                  A Piece of Sky       Fiction       5    5988.18
                    Future Shock   Young Adult       5    5871.08
           The Girl on the Train       Fiction       5    5804.61
                 No Love Allowed   Young Adult       5    5765.58
                Library of Souls   Young Adult       5    5123.08
   Will Grayson Will Grayson ...   Young Adult       5    4991.20
              Until Friday Night   Young Adult       5    4885.70
      The Natural History of ...   Young Adult       5    4770.71
                     The New Guy   Young Adult       5    4739.06
```

## pandas.read_sql and pandas.merge equivalence

`pd.read_sql_query` result for the JOIN:

```text
                           title category_name  rating  price_inr
Aristotle and Dante Discover ...   Young Adult       5    6133.77
                  A Piece of Sky       Fiction       5    5988.18
                    Future Shock   Young Adult       5    5871.08
           The Girl on the Train       Fiction       5    5804.61
                 No Love Allowed   Young Adult       5    5765.58
                Library of Souls   Young Adult       5    5123.08
   Will Grayson Will Grayson ...   Young Adult       5    4991.20
              Until Friday Night   Young Adult       5    4885.70
      The Natural History of ...   Young Adult       5    4770.71
                     The New Guy   Young Adult       5    4739.06
```

`pd.merge` reproduction:

```text
                           title category_name  rating  price_inr
Aristotle and Dante Discover ...   Young Adult       5    6133.77
                  A Piece of Sky       Fiction       5    5988.18
                    Future Shock   Young Adult       5    5871.08
           The Girl on the Train       Fiction       5    5804.61
                 No Love Allowed   Young Adult       5    5765.58
                Library of Souls   Young Adult       5    5123.08
   Will Grayson Will Grayson ...   Young Adult       5    4991.20
              Until Friday Night   Young Adult       5    4885.70
      The Natural History of ...   Young Adult       5    4770.71
                     The New Guy   Young Adult       5    4739.06
```

**Equivalent outputs:** `True`
