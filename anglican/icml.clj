;; gorilla-repl.fileformat = 1

;; **
;;; # Anglican Experiments
;;; 
;;; We define and run the models described in Section 5 in Anglican. Hit `Shift+Enter` to evaluate each code segment.
;; **

;; **
;;; ## Helper functions
;;; 
;;; We start our experiments by defining some helper functions and loading `training_data.txt`.
;; **

;; @@
(ns icml
  (:require [anglican.state :as state]
            [clojure.string :as str]
            :reload)
  (:use clojure.repl
        [anglican core runtime emit]))

; run model with arg for n times
(defn model-multi-run [model arg infs id number burnin n]
  (doall
    (map (fn [alg] (do
                     (println (name alg))
                     (def k (atom 0))
                     (while (< @k n)
                       (do
                         (spit (str/join ["samples/" id "/" (name alg) (str @k) ".txt"]) (time (with-out-str (pr (map :result (take number (drop burnin (doquery alg model [arg]))))))))
                         (swap! k inc)))))
         infs)))

; pretty print times
(defn pretty-print [time-str]
  (print (str/replace (str/replace time-str #"\"Elapsed time: " "") #"\"" "")))

; load training_data for gmm and dpmm
(defn getfloats [string]
  (map #(Float/parseFloat %1)
     (str/split string #",")))
(def training-data (map getfloats (str/split (slurp "training_data.txt") #"\n")))
;; @@

;; **
;;; ## Models
;;; 
;;; The geometric, random walk, Gaussian mixture and Dirichlet process mixture models are defined as below.
;; **

;; @@
; geometric
(defquery geometric
  (loop [n 1]
    (if (< (sample (uniform-continuous 0 1)) 0.2)
      n
      (recur (+ n 1)))))

; random walk
(defquery random-walk
  (let [start (sample (uniform-continuous 0 3))]
    (loop [position start
           distance 0]
      (if (and (> position 0) (< distance 10))
        (let [step (sample (uniform-continuous -1 1))]
          (recur (+ position step)
                 (+ distance (abs step))))
        (observe (normal 1.1 0.1) distance)))
    start))

; gmm
(defdist lik-dist
  [mus std-scalar]
  []
  (sample* [this] nil)
  (observe* [this y]
            (reduce log-sum-exp
                    (map #(- (sum [(observe* (normal (nth %1 0) std-scalar) (nth y 0))
                                   (observe* (normal (nth %1 1) std-scalar) (nth y 1))
                                   (observe* (normal (nth %1 2) std-scalar) (nth y 2))])
                             (log (count mus)))
                         mus))))

(with-primitive-procedures [lik-dist]
  (defquery gmm [data]
    (let [K (+ 1 (sample (poisson 10)))
          mus (loop [k 0
                     mus []]
                (if (= k K)
                  mus
                  (let [mu1 (sample (uniform-continuous 0 100))
                        mu2 (sample (uniform-continuous 0 100))
                        mu3 (sample (uniform-continuous 0 100))]
                    (recur (inc k)
                           (conj mus [mu1 mu2 mu3])))))]
      (map #(observe (lik-dist mus 10) %1) data)
      mus)))

; dpmm
(defdist dp-dist
  [weights means]
  []
  (sample* [this] nil)
  (observe* [this y]
            (reduce log-sum-exp
                    (map #(+ (sum [(observe* (normal (nth %2 0) 10) (nth y 0))
                                   (observe* (normal (nth %2 1) 10) (nth y 1))
                                   (observe* (normal (nth %2 2) 10) (nth y 2))])
                             (log %1))
                         weights means))))

(with-primitive-procedures [dp-dist]
  (defquery dpmm [data]
    (let [dists (loop [cumprod 1
                       beta-val 0
                       stick 1
                       weights []
                       means []]
                  (if (< stick 0.01)
                    [weights means]
                    (let [newcumprod (* cumprod (- 1 beta-val))
                          newbeta (sample (beta 1 5))]
                      (recur newcumprod
                             newbeta
                             (- stick (* newbeta newcumprod))
                             (conj weights (* newbeta newcumprod))
                             (conj means (repeatedly 3 #(sample (uniform-continuous 0 100))))))))
          weights (first dists)
          means (second dists)]
      (map #(observe (dp-dist weights means) %1) data)
      (list weights means))))
;; @@

;; **
;;; ## Run
;;; 
;;; We run our models below where samples will be stored in the `samples/` directory and the elapsed times for each run are printed below.
;; **

;; @@
(println "10 runs of geometric with 5000 samples and 500 burn-in")
(pretty-print (with-out-str (model-multi-run geometric [] [:lmh :rmh :pgibbs :ipmcmc] "geo" 5000 500 10)))

(println "10 runs of random walk with 50000 samples and 5000 burn-in")
(pretty-print (with-out-str (model-multi-run random-walk [] [:lmh :rmh :pgibbs :ipmcmc] "rw" 50000 5000 10)))

(println "10 runs of gmm with 50000 samples and 5000 burn-in")
(pretty-print (with-out-str (model-multi-run gmm training-data [:lmh :rmh :pgibbs :ipmcmc] "gmm" 50000 5000 10)))

(println "10 runs of dpmm with 2000 samples and 1000 burn-in")
(pretty-print (with-out-str (model-multi-run dpmm training-data [:lmh :rmh :pgibbs :ipmcmc] "dpmm" 2000 1000 10)))
;; @@
