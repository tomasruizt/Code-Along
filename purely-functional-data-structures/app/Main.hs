module Main where

suffixes :: [a] -> [[a]]
suffixes [] = [[]]
suffixes whole@(_:xs) = whole : suffixes xs

main :: IO ()
main = print $ suffixes ([1, 2, 3, 4] :: [Integer])

