use std::{
    cmp::Ordering,
    collections::{VecDeque, vec_deque},
    iter::{Chain, FusedIterator, Product, Sum},
    option,
};

use crate::{Peek, PeekBack, PeekNth, PeekNthBack};

use delegate::delegate;

/// An extension trait that allows a copyable iterator to peek ahead at elements via an adapter.
pub trait MakePeekableWithCopy: Iterator + Copy {
    /// Makes a generic copyable iterator peekable. This is generally not that useful, as [std iterators are generally not copyable](https://stackoverflow.com/questions/32324251/why-are-iterators-not-copyable).
    fn peekable_with_copy(self) -> PeekableWithCopy<Self> {
        PeekableWithCopy(self)
    }
}

impl<I: Iterator + Copy> MakePeekableWithCopy for I {}

/// An extension trait that allows a cloneable iterator to peek ahead at elements via an adapter.
pub trait MakePeekableWithClone: Iterator + Clone {
    /// Make a generic cloneable iterator peekable. The [`Peek::peek`] implementation clones the
    /// iterator before calling `next`, so if the iterator has a slow clone, this may not be
    /// advisable.
    ///
    /// Calling this method is generally not required, because std iterators with cheap clone
    /// methods generally have Peekable implemented directly on them, without the requirement for an
    /// adapter.
    fn peekable_with_clone(self) -> PeekableWithClone<Self> {
        PeekableWithClone(self)
    }
}

impl<I: Iterator + Clone> MakePeekableWithClone for I {}

/// An extension trait that allows a iterator to peek ahead at references to elements via an
/// adapter.
pub trait MakePeekableWithBuffer: Iterator + Sized {
    /// Make a generic iterator peekable, arbitrarily deep. Stores peeked but unconsumed elements on
    /// the heap in a [`VecDeque`].
    fn peekable_with_buffer(self) -> PeekableWithBuffer<Self> {
        PeekableWithBuffer {
            inner: self,
            buffer: VecDeque::new(),
        }
    }
}

impl<I: Iterator> MakePeekableWithBuffer for I {}

/// An extension trait that allows an iterator to peek at a reference to the next element via an
/// adapter.
pub trait MakePeekableWithOption: Iterator + Sized {
    /// Make a generic iterator peekable, but only the next element.
    ///
    /// Very similar to
    /// [`Iterator::peekable`], except where [`std::iter::Peekable`] stores an
    /// `Option<Option<Item>>`, `PeekableWithOption` only stores an `Option<Item>`. The upshot of
    /// this is that peeking twice may poll the underlying iterator twice if it is empty, and
    /// calling `peek` and then `next` may return different results on non-fused empty iterators.
    fn peekable_with_option(self) -> PeekableWithOption<Self> {
        PeekableWithOption {
            inner: self,
            peeked: None,
        }
    }
}

impl<I: Iterator> MakePeekableWithOption for I {}

/// An extension trait that allows an iterator to peek at references to the elements at both ends
/// via an adapter.
pub trait MakePeekableBackWithOption: Iterator + Sized {
    /// Make a generic iterator peekable, both from the front and the back, but only the next
    /// and next back elements.
    ///
    /// Like [`PeekableWithOption`] `PeekableBackWithOption` only stores an `Option<Item>` (in
    /// `PeekableBackWithOption`'s case, two `Option<Item>`s). The upshot of this is that
    /// peeking twice may poll the underlying iterator twice if it is empty, and calling `peek`
    /// and then `next` may return different results on non-fused empty iterators.
    fn peekable_back_with_option(self) -> PeekableBackWithOption<Self> {
        PeekableBackWithOption {
            inner: self,
            front: None,
            back: None,
        }
    }
}

impl<I: Iterator> MakePeekableBackWithOption for I {}

/// An extension trait that allows an iterator to peek ahead at references to elements from both
/// ends via an adapter.
pub trait MakePeekableBackWithBuffer: Iterator + Sized {
    /// Make a generic iterator peekable, arbitrarily deep, from both the front and the back. Stores
    /// peeked but unconsumed elements on the heap in two [`VecDeque`]s.
    fn peekable_back_with_buffer(self) -> PeekableBackWithBuffer<Self> {
        PeekableBackWithBuffer {
            inner: self,
            front: VecDeque::new(),
            back: VecDeque::new(),
        }
    }
}

impl<I: Iterator> MakePeekableBackWithBuffer for I {}

/// Peek ahead on copyable iterators. See [`MakePeekableWithCopy::peekable_with_copy`].
#[derive(Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct PeekableWithCopy<I>(I);

/// Peek ahead on cloneable iterators, such as Range. See
/// [`MakePeekableWithClone::peekable_with_clone`].
#[derive(Debug, Clone)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct PeekableWithClone<I>(I);

/// Peek ahead on iterators using a VecDeque buffer to store peeked but yet unconsumed items. See
/// [`MakePeekableWithBuffer::peekable_with_buffer`].
#[derive(Debug, Clone)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct PeekableWithBuffer<I: Iterator> {
    inner: I,
    buffer: VecDeque<I::Item>,
}

/// Peek at the next element of an iterator using an Option to store the peeked but yet unconsumed
/// item. See [`MakePeekableWithOption::peekable_with_option`].
///
/// Unlike [`std::iter::Peekable`], this does not remember that the inner iterator ran dry, so
/// peeking an exhausted iterator always polls it again.
#[derive(Debug, Clone)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct PeekableWithOption<I: Iterator> {
    inner: I,
    peeked: Option<I::Item>,
}

/// Peek at both ends of an iterator using an Option per end to store peeked but yet unconsumed
/// items. See [`MakePeekableBackWithOption::peekable_back_with_option`].
#[derive(Debug, Clone)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct PeekableBackWithOption<I: Iterator> {
    inner: I,
    front: Option<I::Item>,
    back: Option<I::Item>,
}

/// Peek ahead from both ends of an iterator using a VecDeque buffer per end to store peeked but yet
/// unconsumed items. See [`MakePeekableBackWithBuffer::peekable_back_with_buffer`].
#[derive(Debug, Clone)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct PeekableBackWithBuffer<I: Iterator> {
    inner: I,
    front: VecDeque<I::Item>,
    back: VecDeque<I::Item>,
}

impl<I: Copy> Clone for PeekableWithCopy<I> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl<I: Iterator + Copy> Peek for PeekableWithCopy<I> {
    type PeekItem<'a>
        = I::Item
    where
        Self: 'a;

    fn peek<'a>(&'a mut self) -> Option<Self::PeekItem<'a>> {
        let mut copy = self.0;

        copy.next()
    }
}

impl<I: Iterator + Copy> PeekNth for PeekableWithCopy<I> {
    fn peek_nth<'a>(&'a mut self, n: usize) -> Option<Self::PeekItem<'a>> {
        let mut copy = self.0;

        copy.nth(n)
    }
}

impl<I: Iterator + Clone> Peek for PeekableWithClone<I> {
    type PeekItem<'a>
        = I::Item
    where
        Self: 'a;

    fn peek<'a>(&'a mut self) -> Option<Self::PeekItem<'a>> {
        self.0.clone().next()
    }
}

impl<I: Iterator + Clone> PeekNth for PeekableWithClone<I> {
    fn peek_nth<'a>(&'a mut self, n: usize) -> Option<Self::PeekItem<'a>> {
        self.0.clone().nth(n)
    }
}

impl<I: Iterator> Peek for PeekableWithBuffer<I> {
    type PeekItem<'a>
        = &'a mut I::Item
    where
        Self: 'a;

    fn peek<'a>(&'a mut self) -> Option<Self::PeekItem<'a>> {
        if self.buffer.is_empty() {
            let item = self.inner.next()?;
            self.buffer.push_back(item);
        }

        self.buffer.front_mut()
    }
}

impl<I: Iterator> PeekNth for PeekableWithBuffer<I> {
    fn peek_nth<'a>(&'a mut self, n: usize) -> Option<Self::PeekItem<'a>> {
        assert!(n < usize::MAX);
        let missing = n.saturating_add(1).saturating_sub(self.buffer.len());
        self.buffer.extend((&mut self.inner).take(missing));

        self.buffer.get_mut(n)
    }
}

impl<I: Iterator> Peek for PeekableWithOption<I> {
    type PeekItem<'a>
        = &'a mut I::Item
    where
        Self: 'a;

    fn peek<'a>(&'a mut self) -> Option<Self::PeekItem<'a>> {
        // Caching exhaustion would take an `Option<Option<_>>`, so an empty slot just polls again.
        if self.peeked.is_none() {
            self.peeked = self.inner.next();
        }

        self.peeked.as_mut()
    }
}

impl<I: Iterator> Peek for PeekableBackWithOption<I> {
    type PeekItem<'a>
        = &'a mut I::Item
    where
        Self: 'a;

    fn peek<'a>(&'a mut self) -> Option<Self::PeekItem<'a>> {
        // Once the inner iterator runs dry, the back slot holds the next item in sequence.
        if self.front.is_none() {
            self.front = self.inner.next().or_else(|| self.back.take());
        }

        self.front.as_mut()
    }
}

impl<I: DoubleEndedIterator> PeekBack for PeekableBackWithOption<I> {
    fn peek_back<'a>(&'a mut self) -> Option<Self::PeekItem<'a>> {
        // Mirror of `peek`: the front slot holds the last item once the inner iterator runs dry.
        if self.back.is_none() {
            self.back = self.inner.next_back().or_else(|| self.front.take());
        }

        self.back.as_mut()
    }
}

impl<I: Iterator> Peek for PeekableBackWithBuffer<I> {
    type PeekItem<'a>
        = &'a mut I::Item
    where
        Self: 'a;

    fn peek<'a>(&'a mut self) -> Option<Self::PeekItem<'a>> {
        self.peek_nth(0)
    }
}

impl<I: Iterator> PeekNth for PeekableBackWithBuffer<I> {
    fn peek_nth<'a>(&'a mut self, n: usize) -> Option<Self::PeekItem<'a>> {
        assert!(n < usize::MAX);
        while self.front.len() <= n {
            let item = self.inner.next().or_else(|| self.back.pop_front())?;
            self.front.push_back(item);
        }

        self.front.get_mut(n)
    }
}

impl<I: DoubleEndedIterator> PeekBack for PeekableBackWithBuffer<I> {
    fn peek_back<'a>(&'a mut self) -> Option<Self::PeekItem<'a>> {
        self.peek_nth_back(0)
    }
}

impl<I: DoubleEndedIterator> PeekNthBack for PeekableBackWithBuffer<I> {
    fn peek_nth_back<'a>(&'a mut self, n: usize) -> Option<Self::PeekItem<'a>> {
        assert!(n < usize::MAX);
        self.back.reserve(n.saturating_sub(self.back.len()));
        while self.back.len() <= n {
            let item = self.inner.next_back().or_else(|| self.front.pop_back())?;
            self.back.push_front(item);
        }

        let last = self.back.len() - 1;
        self.back.get_mut(last - n)
    }
}

impl<I: Iterator> Iterator for PeekableWithCopy<I> {
    type Item = I::Item;

    delegate! {
        to self.0 {
            fn next(&mut self) -> Option<Self::Item>;
            fn size_hint(&self) -> (usize, Option<usize>);
            fn count(self) -> usize;
            fn last(self) -> Option<Self::Item>;
            fn nth(&mut self, n: usize) -> Option<Self::Item>;

            fn fold<B, F>(self, init: B, f: F) -> B
            where F: FnMut(B, Self::Item) -> B;
            fn for_each<F>(self, f: F)
            where F: FnMut(Self::Item);
            fn reduce<F>(self, f: F) -> Option<Self::Item>
            where F: FnMut(Self::Item, Self::Item) -> Self::Item;

            fn all<F>(&mut self, f: F) -> bool
            where F: FnMut(Self::Item) -> bool;
            fn any<F>(&mut self, f: F) -> bool
            where F: FnMut(Self::Item) -> bool;
            fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
            where P: FnMut(&Self::Item) -> bool;
            fn find_map<B, F>(&mut self, f: F) -> Option<B>
            where F: FnMut(Self::Item) -> Option<B>;
            fn position<P>(&mut self, predicate: P) -> Option<usize>
            where P: FnMut(Self::Item) -> bool;

            fn max(self) -> Option<Self::Item>
            where Self::Item: Ord;
            fn min(self) -> Option<Self::Item>
            where Self::Item: Ord;
            fn max_by<F>(self, compare: F) -> Option<Self::Item>
            where F: FnMut(&Self::Item, &Self::Item) -> Ordering;
            fn min_by<F>(self, compare: F) -> Option<Self::Item>
            where F: FnMut(&Self::Item, &Self::Item) -> Ordering;
            fn max_by_key<B, F>(self, f: F) -> Option<Self::Item>
            where B: Ord, F: FnMut(&Self::Item) -> B;
            fn min_by_key<B, F>(self, f: F) -> Option<Self::Item>
            where B: Ord, F: FnMut(&Self::Item) -> B;

            fn sum<S>(self) -> S
            where S: Sum<Self::Item>;
            fn product<P>(self) -> P
            where P: Product<Self::Item>;

            fn partition<B, F>(self, f: F) -> (B, B)
            where B: Default + Extend<Self::Item>, F: FnMut(&Self::Item) -> bool;

            fn cmp<J>(self, other: J) -> Ordering
            where J: IntoIterator<Item = Self::Item>, Self::Item: Ord;
            fn partial_cmp<J>(self, other: J) -> Option<Ordering>
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn eq<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialEq<J::Item>;
            fn ne<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialEq<J::Item>;
            fn lt<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn le<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn gt<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn ge<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;

            fn is_sorted(self) -> bool
            where Self::Item: PartialOrd;
            fn is_sorted_by<F>(self, compare: F) -> bool
            where F: FnMut(&Self::Item, &Self::Item) -> bool;
            fn is_sorted_by_key<F, K>(self, f: F) -> bool
            where F: FnMut(Self::Item) -> K, K: PartialOrd;
        }
    }
}
impl<I: DoubleEndedIterator> DoubleEndedIterator for PeekableWithCopy<I> {
    delegate! {
        to self.0 {
            fn next_back(&mut self) -> Option<Self::Item>;
            fn nth_back(&mut self, n: usize) -> Option<Self::Item>;
            fn rfold<B, F>(self, init: B, f: F) -> B
            where F: FnMut(B, Self::Item) -> B;
            fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
            where P: FnMut(&Self::Item) -> bool;
        }
    }
}
impl<I: ExactSizeIterator> ExactSizeIterator for PeekableWithCopy<I> {
    delegate! {
        to self.0 {
            fn len(&self) -> usize;
        }
    }
}
impl<I: FusedIterator> FusedIterator for PeekableWithCopy<I> {}

impl<I: Iterator> Iterator for PeekableWithClone<I> {
    type Item = I::Item;

    delegate! {
        to self.0 {
            fn next(&mut self) -> Option<Self::Item>;
            fn size_hint(&self) -> (usize, Option<usize>);
            fn count(self) -> usize;
            fn last(self) -> Option<Self::Item>;
            fn nth(&mut self, n: usize) -> Option<Self::Item>;

            fn fold<B, F>(self, init: B, f: F) -> B
            where F: FnMut(B, Self::Item) -> B;
            fn for_each<F>(self, f: F)
            where F: FnMut(Self::Item);
            fn reduce<F>(self, f: F) -> Option<Self::Item>
            where F: FnMut(Self::Item, Self::Item) -> Self::Item;

            fn all<F>(&mut self, f: F) -> bool
            where F: FnMut(Self::Item) -> bool;
            fn any<F>(&mut self, f: F) -> bool
            where F: FnMut(Self::Item) -> bool;
            fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
            where P: FnMut(&Self::Item) -> bool;
            fn find_map<B, F>(&mut self, f: F) -> Option<B>
            where F: FnMut(Self::Item) -> Option<B>;
            fn position<P>(&mut self, predicate: P) -> Option<usize>
            where P: FnMut(Self::Item) -> bool;

            fn max(self) -> Option<Self::Item>
            where Self::Item: Ord;
            fn min(self) -> Option<Self::Item>
            where Self::Item: Ord;
            fn max_by<F>(self, compare: F) -> Option<Self::Item>
            where F: FnMut(&Self::Item, &Self::Item) -> Ordering;
            fn min_by<F>(self, compare: F) -> Option<Self::Item>
            where F: FnMut(&Self::Item, &Self::Item) -> Ordering;
            fn max_by_key<B, F>(self, f: F) -> Option<Self::Item>
            where B: Ord, F: FnMut(&Self::Item) -> B;
            fn min_by_key<B, F>(self, f: F) -> Option<Self::Item>
            where B: Ord, F: FnMut(&Self::Item) -> B;

            fn sum<S>(self) -> S
            where S: Sum<Self::Item>;
            fn product<P>(self) -> P
            where P: Product<Self::Item>;

            fn partition<B, F>(self, f: F) -> (B, B)
            where B: Default + Extend<Self::Item>, F: FnMut(&Self::Item) -> bool;

            fn cmp<J>(self, other: J) -> Ordering
            where J: IntoIterator<Item = Self::Item>, Self::Item: Ord;
            fn partial_cmp<J>(self, other: J) -> Option<Ordering>
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn eq<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialEq<J::Item>;
            fn ne<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialEq<J::Item>;
            fn lt<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn le<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn gt<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn ge<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;

            fn is_sorted(self) -> bool
            where Self::Item: PartialOrd;
            fn is_sorted_by<F>(self, compare: F) -> bool
            where F: FnMut(&Self::Item, &Self::Item) -> bool;
            fn is_sorted_by_key<F, K>(self, f: F) -> bool
            where F: FnMut(Self::Item) -> K, K: PartialOrd;
        }
    }
}
impl<I: DoubleEndedIterator> DoubleEndedIterator for PeekableWithClone<I> {
    delegate! {
        to self.0 {
            fn next_back(&mut self) -> Option<Self::Item>;
            fn nth_back(&mut self, n: usize) -> Option<Self::Item>;
            fn rfold<B, F>(self, init: B, f: F) -> B
            where F: FnMut(B, Self::Item) -> B;
            fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
            where P: FnMut(&Self::Item) -> bool;
        }
    }
}
impl<I: ExactSizeIterator> ExactSizeIterator for PeekableWithClone<I> {
    delegate! {
        to self.0 {
            fn len(&self) -> usize;
        }
    }
}
impl<I: FusedIterator> FusedIterator for PeekableWithClone<I> {}

// Iterates a `VecDeque` by popping, so early exit leaves the unconsumed items in the buffer (unlike
// `VecDeque::drain`, which removes them on drop).
struct PopBuffer<'a, T>(&'a mut VecDeque<T>);

impl<T> Iterator for PopBuffer<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.0.pop_front()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.len();
        (n, Some(n))
    }
}

impl<T> DoubleEndedIterator for PopBuffer<'_, T> {
    fn next_back(&mut self) -> Option<T> {
        self.0.pop_back()
    }
}

impl<T> ExactSizeIterator for PopBuffer<'_, T> {}
impl<T> FusedIterator for PopBuffer<'_, T> {}

// The single-slot counterpart of `PopBuffer`: the item only leaves the slot once it is yielded.
struct TakeOption<'a, T>(&'a mut Option<T>);

impl<T> Iterator for TakeOption<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.0.take()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = usize::from(self.0.is_some());
        (n, Some(n))
    }
}

impl<T> DoubleEndedIterator for TakeOption<'_, T> {
    fn next_back(&mut self) -> Option<T> {
        self.0.take()
    }
}

impl<T> ExactSizeIterator for TakeOption<'_, T> {}
impl<T> FusedIterator for TakeOption<'_, T> {}

impl<I: Iterator> PeekableWithBuffer<I> {
    fn into_chain(self) -> Chain<vec_deque::IntoIter<I::Item>, I> {
        self.buffer.into_iter().chain(self.inner)
    }

    fn as_mut_chain(&mut self) -> Chain<PopBuffer<'_, I::Item>, &mut I> {
        PopBuffer(&mut self.buffer).chain(&mut self.inner)
    }
}

impl<I: Iterator> Iterator for PeekableWithBuffer<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.buffer.pop_front().or_else(|| self.inner.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.buffer.len();
        let (lo, hi) = self.inner.size_hint();
        (lo.saturating_add(n), hi.and_then(|hi| hi.checked_add(n)))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let buffered = self.buffer.len();
        if n < buffered {
            drop(self.buffer.drain(..n));
            self.buffer.pop_front()
        } else {
            self.buffer.clear();
            self.inner.nth(n - buffered)
        }
    }

    delegate! {
        to self.into_chain() {
            fn count(self) -> usize;
            fn last(self) -> Option<Self::Item>;

            fn fold<B, F>(self, init: B, f: F) -> B
            where F: FnMut(B, Self::Item) -> B;
            fn for_each<F>(self, f: F)
            where F: FnMut(Self::Item);
            fn reduce<F>(self, f: F) -> Option<Self::Item>
            where F: FnMut(Self::Item, Self::Item) -> Self::Item;

            fn max(self) -> Option<Self::Item>
            where Self::Item: Ord;
            fn min(self) -> Option<Self::Item>
            where Self::Item: Ord;
            fn max_by<F>(self, compare: F) -> Option<Self::Item>
            where F: FnMut(&Self::Item, &Self::Item) -> Ordering;
            fn min_by<F>(self, compare: F) -> Option<Self::Item>
            where F: FnMut(&Self::Item, &Self::Item) -> Ordering;
            fn max_by_key<B, F>(self, f: F) -> Option<Self::Item>
            where B: Ord, F: FnMut(&Self::Item) -> B;
            fn min_by_key<B, F>(self, f: F) -> Option<Self::Item>
            where B: Ord, F: FnMut(&Self::Item) -> B;

            fn sum<S>(self) -> S
            where S: Sum<Self::Item>;
            fn product<P>(self) -> P
            where P: Product<Self::Item>;

            fn partition<B, F>(self, f: F) -> (B, B)
            where B: Default + Extend<Self::Item>, F: FnMut(&Self::Item) -> bool;

            fn cmp<J>(self, other: J) -> Ordering
            where J: IntoIterator<Item = Self::Item>, Self::Item: Ord;
            fn partial_cmp<J>(self, other: J) -> Option<Ordering>
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn eq<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialEq<J::Item>;
            fn ne<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialEq<J::Item>;
            fn lt<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn le<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn gt<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn ge<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;

            fn is_sorted(self) -> bool
            where Self::Item: PartialOrd;
            fn is_sorted_by<F>(self, compare: F) -> bool
            where F: FnMut(&Self::Item, &Self::Item) -> bool;
            fn is_sorted_by_key<F, K>(self, f: F) -> bool
            where F: FnMut(Self::Item) -> K, K: PartialOrd;
        }

        to self.as_mut_chain() {
            fn all<F>(&mut self, f: F) -> bool
            where F: FnMut(Self::Item) -> bool;
            fn any<F>(&mut self, f: F) -> bool
            where F: FnMut(Self::Item) -> bool;
            fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
            where P: FnMut(&Self::Item) -> bool;
            fn find_map<B, F>(&mut self, f: F) -> Option<B>
            where F: FnMut(Self::Item) -> Option<B>;
            fn position<P>(&mut self, predicate: P) -> Option<usize>
            where P: FnMut(Self::Item) -> bool;
        }
    }
}

impl<I: DoubleEndedIterator> DoubleEndedIterator for PeekableWithBuffer<I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().or_else(|| self.buffer.pop_back())
    }

    delegate! {
        to self.into_chain() {
            fn rfold<B, F>(self, init: B, f: F) -> B
            where F: FnMut(B, Self::Item) -> B;
        }

        to self.as_mut_chain() {
            fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
            where P: FnMut(&Self::Item) -> bool;
        }
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for PeekableWithBuffer<I> {
    fn len(&self) -> usize {
        self.buffer.len() + self.inner.len()
    }
}

impl<I: FusedIterator> FusedIterator for PeekableWithBuffer<I> {}

impl<I: Iterator> PeekableWithOption<I> {
    fn into_chain(self) -> Chain<option::IntoIter<I::Item>, I> {
        self.peeked.into_iter().chain(self.inner)
    }

    fn as_mut_chain(&mut self) -> Chain<TakeOption<'_, I::Item>, &mut I> {
        TakeOption(&mut self.peeked).chain(&mut self.inner)
    }
}

impl<I: Iterator> Iterator for PeekableWithOption<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.peeked.take().or_else(|| self.inner.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = usize::from(self.peeked.is_some());
        let (lo, hi) = self.inner.size_hint();
        (lo.saturating_add(n), hi.and_then(|hi| hi.checked_add(n)))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match self.peeked.take() {
            Some(item) if n == 0 => Some(item),
            Some(_) => self.inner.nth(n - 1),
            None => self.inner.nth(n),
        }
    }

    delegate! {
        to self.into_chain() {
            fn count(self) -> usize;
            fn last(self) -> Option<Self::Item>;

            fn fold<B, F>(self, init: B, f: F) -> B
            where F: FnMut(B, Self::Item) -> B;
            fn for_each<F>(self, f: F)
            where F: FnMut(Self::Item);
            fn reduce<F>(self, f: F) -> Option<Self::Item>
            where F: FnMut(Self::Item, Self::Item) -> Self::Item;

            fn max(self) -> Option<Self::Item>
            where Self::Item: Ord;
            fn min(self) -> Option<Self::Item>
            where Self::Item: Ord;
            fn max_by<F>(self, compare: F) -> Option<Self::Item>
            where F: FnMut(&Self::Item, &Self::Item) -> Ordering;
            fn min_by<F>(self, compare: F) -> Option<Self::Item>
            where F: FnMut(&Self::Item, &Self::Item) -> Ordering;
            fn max_by_key<B, F>(self, f: F) -> Option<Self::Item>
            where B: Ord, F: FnMut(&Self::Item) -> B;
            fn min_by_key<B, F>(self, f: F) -> Option<Self::Item>
            where B: Ord, F: FnMut(&Self::Item) -> B;

            fn sum<S>(self) -> S
            where S: Sum<Self::Item>;
            fn product<P>(self) -> P
            where P: Product<Self::Item>;

            fn partition<B, F>(self, f: F) -> (B, B)
            where B: Default + Extend<Self::Item>, F: FnMut(&Self::Item) -> bool;

            fn cmp<J>(self, other: J) -> Ordering
            where J: IntoIterator<Item = Self::Item>, Self::Item: Ord;
            fn partial_cmp<J>(self, other: J) -> Option<Ordering>
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn eq<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialEq<J::Item>;
            fn ne<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialEq<J::Item>;
            fn lt<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn le<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn gt<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn ge<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;

            fn is_sorted(self) -> bool
            where Self::Item: PartialOrd;
            fn is_sorted_by<F>(self, compare: F) -> bool
            where F: FnMut(&Self::Item, &Self::Item) -> bool;
            fn is_sorted_by_key<F, K>(self, f: F) -> bool
            where F: FnMut(Self::Item) -> K, K: PartialOrd;
        }

        to self.as_mut_chain() {
            fn all<F>(&mut self, f: F) -> bool
            where F: FnMut(Self::Item) -> bool;
            fn any<F>(&mut self, f: F) -> bool
            where F: FnMut(Self::Item) -> bool;
            fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
            where P: FnMut(&Self::Item) -> bool;
            fn find_map<B, F>(&mut self, f: F) -> Option<B>
            where F: FnMut(Self::Item) -> Option<B>;
            fn position<P>(&mut self, predicate: P) -> Option<usize>
            where P: FnMut(Self::Item) -> bool;
        }
    }
}

impl<I: DoubleEndedIterator> DoubleEndedIterator for PeekableWithOption<I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().or_else(|| self.peeked.take())
    }

    delegate! {
        to self.into_chain() {
            fn rfold<B, F>(self, init: B, f: F) -> B
            where F: FnMut(B, Self::Item) -> B;
        }

        to self.as_mut_chain() {
            fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
            where P: FnMut(&Self::Item) -> bool;
        }
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for PeekableWithOption<I> {
    fn len(&self) -> usize {
        usize::from(self.peeked.is_some()) + self.inner.len()
    }
}

impl<I: FusedIterator> FusedIterator for PeekableWithOption<I> {}

impl<I: Iterator> PeekableBackWithOption<I> {
    fn into_chain(self) -> Chain<Chain<option::IntoIter<I::Item>, I>, option::IntoIter<I::Item>> {
        self.front.into_iter().chain(self.inner).chain(self.back)
    }

    fn as_mut_chain(
        &mut self,
    ) -> Chain<Chain<TakeOption<'_, I::Item>, &mut I>, TakeOption<'_, I::Item>> {
        TakeOption(&mut self.front)
            .chain(&mut self.inner)
            .chain(TakeOption(&mut self.back))
    }
}

impl<I: Iterator> Iterator for PeekableBackWithOption<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.front
            .take()
            .or_else(|| self.inner.next())
            .or_else(|| self.back.take())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = usize::from(self.front.is_some()) + usize::from(self.back.is_some());
        let (lo, hi) = self.inner.size_hint();
        (lo.saturating_add(n), hi.and_then(|hi| hi.checked_add(n)))
    }

    fn nth(&mut self, mut n: usize) -> Option<Self::Item> {
        if let Some(item) = self.front.take() {
            if n == 0 {
                return Some(item);
            }
            n -= 1;
        }

        // `inner.nth` can't report how far it got before running dry, and that decides whether the
        // back slot is skipped too.
        let skipped = (&mut self.inner).take(n).count();
        if skipped < n {
            self.back = None;
            return None;
        }

        self.inner.next().or_else(|| self.back.take())
    }

    delegate! {
        to self.into_chain() {
            fn count(self) -> usize;
            fn last(self) -> Option<Self::Item>;

            fn fold<B, F>(self, init: B, f: F) -> B
            where F: FnMut(B, Self::Item) -> B;
            fn for_each<F>(self, f: F)
            where F: FnMut(Self::Item);
            fn reduce<F>(self, f: F) -> Option<Self::Item>
            where F: FnMut(Self::Item, Self::Item) -> Self::Item;

            fn max(self) -> Option<Self::Item>
            where Self::Item: Ord;
            fn min(self) -> Option<Self::Item>
            where Self::Item: Ord;
            fn max_by<F>(self, compare: F) -> Option<Self::Item>
            where F: FnMut(&Self::Item, &Self::Item) -> Ordering;
            fn min_by<F>(self, compare: F) -> Option<Self::Item>
            where F: FnMut(&Self::Item, &Self::Item) -> Ordering;
            fn max_by_key<B, F>(self, f: F) -> Option<Self::Item>
            where B: Ord, F: FnMut(&Self::Item) -> B;
            fn min_by_key<B, F>(self, f: F) -> Option<Self::Item>
            where B: Ord, F: FnMut(&Self::Item) -> B;

            fn sum<S>(self) -> S
            where S: Sum<Self::Item>;
            fn product<P>(self) -> P
            where P: Product<Self::Item>;

            fn partition<B, F>(self, f: F) -> (B, B)
            where B: Default + Extend<Self::Item>, F: FnMut(&Self::Item) -> bool;

            fn cmp<J>(self, other: J) -> Ordering
            where J: IntoIterator<Item = Self::Item>, Self::Item: Ord;
            fn partial_cmp<J>(self, other: J) -> Option<Ordering>
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn eq<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialEq<J::Item>;
            fn ne<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialEq<J::Item>;
            fn lt<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn le<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn gt<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn ge<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;

            fn is_sorted(self) -> bool
            where Self::Item: PartialOrd;
            fn is_sorted_by<F>(self, compare: F) -> bool
            where F: FnMut(&Self::Item, &Self::Item) -> bool;
            fn is_sorted_by_key<F, K>(self, f: F) -> bool
            where F: FnMut(Self::Item) -> K, K: PartialOrd;
        }

        to self.as_mut_chain() {
            fn all<F>(&mut self, f: F) -> bool
            where F: FnMut(Self::Item) -> bool;
            fn any<F>(&mut self, f: F) -> bool
            where F: FnMut(Self::Item) -> bool;
            fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
            where P: FnMut(&Self::Item) -> bool;
            fn find_map<B, F>(&mut self, f: F) -> Option<B>
            where F: FnMut(Self::Item) -> Option<B>;
            fn position<P>(&mut self, predicate: P) -> Option<usize>
            where P: FnMut(Self::Item) -> bool;
        }
    }
}

impl<I: DoubleEndedIterator> DoubleEndedIterator for PeekableBackWithOption<I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.back
            .take()
            .or_else(|| self.inner.next_back())
            .or_else(|| self.front.take())
    }

    delegate! {
        to self.into_chain() {
            fn rfold<B, F>(self, init: B, f: F) -> B
            where F: FnMut(B, Self::Item) -> B;
        }

        to self.as_mut_chain() {
            fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
            where P: FnMut(&Self::Item) -> bool;
        }
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for PeekableBackWithOption<I> {
    fn len(&self) -> usize {
        usize::from(self.front.is_some()) + self.inner.len() + usize::from(self.back.is_some())
    }
}

impl<I: FusedIterator> FusedIterator for PeekableBackWithOption<I> {}

impl<I: Iterator> PeekableBackWithBuffer<I> {
    fn into_chain(
        self,
    ) -> Chain<Chain<vec_deque::IntoIter<I::Item>, I>, vec_deque::IntoIter<I::Item>> {
        self.front.into_iter().chain(self.inner).chain(self.back)
    }

    fn as_mut_chain(
        &mut self,
    ) -> Chain<Chain<PopBuffer<'_, I::Item>, &mut I>, PopBuffer<'_, I::Item>> {
        PopBuffer(&mut self.front)
            .chain(&mut self.inner)
            .chain(PopBuffer(&mut self.back))
    }
}

impl<I: Iterator> Iterator for PeekableBackWithBuffer<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.front
            .pop_front()
            .or_else(|| self.inner.next())
            .or_else(|| self.back.pop_front())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.front.len() + self.back.len();
        let (lo, hi) = self.inner.size_hint();
        (lo.saturating_add(n), hi.and_then(|hi| hi.checked_add(n)))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let buffered = self.front.len();
        if n < buffered {
            drop(self.front.drain(..n));
            return self.front.pop_front();
        }
        self.front.clear();

        // `inner.nth` can't report how far it got before running dry, and that decides how much of
        // the back buffer is skipped too.
        let remaining = n - buffered;
        let skipped = (&mut self.inner).take(remaining).count();
        if skipped == remaining {
            return self.inner.next().or_else(|| self.back.pop_front());
        }

        let rest = remaining - skipped;
        drop(self.back.drain(..rest.min(self.back.len())));
        self.back.pop_front()
    }

    delegate! {
        to self.into_chain() {
            fn count(self) -> usize;
            fn last(self) -> Option<Self::Item>;

            fn fold<B, F>(self, init: B, f: F) -> B
            where F: FnMut(B, Self::Item) -> B;
            fn for_each<F>(self, f: F)
            where F: FnMut(Self::Item);
            fn reduce<F>(self, f: F) -> Option<Self::Item>
            where F: FnMut(Self::Item, Self::Item) -> Self::Item;

            fn max(self) -> Option<Self::Item>
            where Self::Item: Ord;
            fn min(self) -> Option<Self::Item>
            where Self::Item: Ord;
            fn max_by<F>(self, compare: F) -> Option<Self::Item>
            where F: FnMut(&Self::Item, &Self::Item) -> Ordering;
            fn min_by<F>(self, compare: F) -> Option<Self::Item>
            where F: FnMut(&Self::Item, &Self::Item) -> Ordering;
            fn max_by_key<B, F>(self, f: F) -> Option<Self::Item>
            where B: Ord, F: FnMut(&Self::Item) -> B;
            fn min_by_key<B, F>(self, f: F) -> Option<Self::Item>
            where B: Ord, F: FnMut(&Self::Item) -> B;

            fn sum<S>(self) -> S
            where S: Sum<Self::Item>;
            fn product<P>(self) -> P
            where P: Product<Self::Item>;

            fn partition<B, F>(self, f: F) -> (B, B)
            where B: Default + Extend<Self::Item>, F: FnMut(&Self::Item) -> bool;

            fn cmp<J>(self, other: J) -> Ordering
            where J: IntoIterator<Item = Self::Item>, Self::Item: Ord;
            fn partial_cmp<J>(self, other: J) -> Option<Ordering>
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn eq<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialEq<J::Item>;
            fn ne<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialEq<J::Item>;
            fn lt<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn le<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn gt<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;
            fn ge<J>(self, other: J) -> bool
            where J: IntoIterator, Self::Item: PartialOrd<J::Item>;

            fn is_sorted(self) -> bool
            where Self::Item: PartialOrd;
            fn is_sorted_by<F>(self, compare: F) -> bool
            where F: FnMut(&Self::Item, &Self::Item) -> bool;
            fn is_sorted_by_key<F, K>(self, f: F) -> bool
            where F: FnMut(Self::Item) -> K, K: PartialOrd;
        }

        to self.as_mut_chain() {
            fn all<F>(&mut self, f: F) -> bool
            where F: FnMut(Self::Item) -> bool;
            fn any<F>(&mut self, f: F) -> bool
            where F: FnMut(Self::Item) -> bool;
            fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
            where P: FnMut(&Self::Item) -> bool;
            fn find_map<B, F>(&mut self, f: F) -> Option<B>
            where F: FnMut(Self::Item) -> Option<B>;
            fn position<P>(&mut self, predicate: P) -> Option<usize>
            where P: FnMut(Self::Item) -> bool;
        }
    }
}

impl<I: DoubleEndedIterator> DoubleEndedIterator for PeekableBackWithBuffer<I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.back
            .pop_back()
            .or_else(|| self.inner.next_back())
            .or_else(|| self.front.pop_back())
    }

    delegate! {
        to self.into_chain() {
            fn rfold<B, F>(self, init: B, f: F) -> B
            where F: FnMut(B, Self::Item) -> B;
        }

        to self.as_mut_chain() {
            fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
            where P: FnMut(&Self::Item) -> bool;
        }
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for PeekableBackWithBuffer<I> {
    fn len(&self) -> usize {
        self.front.len() + self.inner.len() + self.back.len()
    }
}

impl<I: FusedIterator> FusedIterator for PeekableBackWithBuffer<I> {}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use std::ops::Range;

    fn assert_fused<T: FusedIterator>(_: &T) {}

    // Replacement for Range that is Copy
    #[derive(Debug, Clone, Copy)]
    struct Span<T = i32> {
        start: T,
        end: T,
    }

    impl<T> From<Range<T>> for Span<T> {
        fn from(range: Range<T>) -> Self {
            Span {
                start: range.start,
                end: range.end,
            }
        }
    }

    impl<T: PartialOrd + std::ops::Add<i32, Output = T> + Copy + std::borrow::Borrow<i32>> Iterator
        for Span<T>
    {
        type Item = T;

        fn next(&mut self) -> Option<T> {
            (self.start < self.end).then(|| {
                self.start = self.start + 1;
                self.start + -1
            })
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let n = (self.end.borrow() - self.start.borrow()).max(0) as usize;
            (n, Some(n))
        }
    }

    impl DoubleEndedIterator for Span {
        fn next_back(&mut self) -> Option<i32> {
            (self.start < self.end).then(|| {
                self.end -= 1;
                self.end
            })
        }
    }

    impl ExactSizeIterator for Span {}
    impl FusedIterator for Span {}

    // Never ends, and counts calls to `next`, wrapping on overflow.
    struct Counter {
        count: usize,
    }

    impl Iterator for Counter {
        type Item = ();

        fn next(&mut self) -> Option<()> {
            self.count = self.count.wrapping_add(1);
            Some(())
        }
    }
    impl DoubleEndedIterator for Counter {
        fn next_back(&mut self) -> Option<()> {
            self.count = self.count.wrapping_add(1);
            Some(())
        }
    }

    // Reports the most extreme size hint possible, to reach the overflow paths.
    struct Hinted;

    impl Iterator for Hinted {
        type Item = u8;

        fn next(&mut self) -> Option<u8> {
            Some(0)
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (usize::MAX, Some(usize::MAX))
        }
    }

    // Yields `None` in the middle of the sequence and then carries on, so it is not fused.
    struct Flaky(std::vec::IntoIter<Option<i32>>);

    impl Iterator for Flaky {
        type Item = i32;

        fn next(&mut self) -> Option<i32> {
            self.0.next().flatten()
        }
    }

    /// How many items to peek from each end before a test runs.
    #[derive(Debug, Clone, Copy)]
    struct Priming {
        front: usize,
        back: usize,
    }

    const fn pr(front: usize, back: usize) -> Priming {
        Priming { front, back }
    }

    trait Fixture: DoubleEndedIterator<Item = i32> + ExactSizeIterator + FusedIterator + Clone {
        const PRIMINGS: &'static [Priming];

        // Primes the Peekable by calling peek_nth(priming.front)/peek() and
        // peek_nth_back(priming.back)/peek_back(), depending on what is available.
        fn primed(range: Range<i32>, priming: Priming) -> Self;
        // The number of currently buffered items.
        fn buffered(&self) -> usize;
    }

    // Runs `check` once per priming, so every consumer sees each buffer/inner boundary.
    fn each<T: Fixture>(mut check: impl FnMut(&dyn Fn(Range<i32>) -> T)) {
        for &priming in T::PRIMINGS {
            // Captured by the harness, so a failure shows which priming it was.
            eprintln!("{priming:?}");
            check(&|range| T::primed(range, priming));
        }
    }

    // Peeking copies, so there is no state to prime.
    impl Fixture for PeekableWithCopy<Span> {
        const PRIMINGS: &'static [Priming] = &[pr(0, 0)];

        fn primed(range: Range<i32>, _: Priming) -> Self {
            Span::from(range).peekable_with_copy()
        }

        fn buffered(&self) -> usize {
            0
        }
    }

    impl Fixture for PeekableWithClone<Range<i32>> {
        const PRIMINGS: &'static [Priming] = &[pr(0, 0)];

        fn primed(range: Range<i32>, _: Priming) -> Self {
            range.peekable_with_clone()
        }

        fn buffered(&self) -> usize {
            0
        }
    }

    impl Fixture for PeekableWithBuffer<Range<i32>> {
        const PRIMINGS: &'static [Priming] = &[pr(0, 0), pr(2, 0), pr(4, 0), pr(6, 0)];

        fn primed(range: Range<i32>, priming: Priming) -> Self {
            let mut p = range.peekable_with_buffer();
            if priming.front > 0 {
                p.peek_nth(priming.front - 1);
            }
            p
        }

        fn buffered(&self) -> usize {
            self.buffer.len()
        }
    }

    impl Fixture for PeekableWithOption<Range<i32>> {
        const PRIMINGS: &'static [Priming] = &[pr(0, 0), pr(1, 0)];

        fn primed(range: Range<i32>, priming: Priming) -> Self {
            let mut p = range.peekable_with_option();
            if priming.front > 0 {
                p.peek();
            }
            p
        }

        fn buffered(&self) -> usize {
            usize::from(self.peeked.is_some())
        }
    }

    impl Fixture for PeekableBackWithOption<Range<i32>> {
        const PRIMINGS: &'static [Priming] = &[pr(0, 0), pr(1, 0), pr(0, 1), pr(1, 1)];

        fn primed(range: Range<i32>, priming: Priming) -> Self {
            let mut p = range.peekable_back_with_option();
            if priming.front > 0 {
                p.peek();
            }
            if priming.back > 0 {
                p.peek_back();
            }
            p
        }

        fn buffered(&self) -> usize {
            usize::from(self.front.is_some()) + usize::from(self.back.is_some())
        }
    }

    impl Fixture for PeekableBackWithBuffer<Range<i32>> {
        const PRIMINGS: &'static [Priming] = &[
            pr(0, 0),
            pr(2, 0),
            pr(0, 2),
            pr(2, 2),
            pr(3, 3),
            pr(6, 0),
            pr(0, 6),
        ];

        fn primed(range: Range<i32>, priming: Priming) -> Self {
            let mut p = range.peekable_back_with_buffer();
            if priming.front > 0 {
                p.peek_nth(priming.front - 1);
            }
            if priming.back > 0 {
                p.peek_nth_back(priming.back - 1);
            }
            p
        }

        fn buffered(&self) -> usize {
            self.front.len() + self.back.len()
        }
    }

    type CopySpan = PeekableWithCopy<Span>;
    type CloneRange = PeekableWithClone<Range<i32>>;
    type BufferRange = PeekableWithBuffer<Range<i32>>;
    type OptionRange = PeekableWithOption<Range<i32>>;
    type BackOptionRange = PeekableBackWithOption<Range<i32>>;
    type BackBufferRange = PeekableBackWithBuffer<Range<i32>>;

    // Peek items are values for the copying types and `&mut` for the buffering ones, so these
    // give the tests one `i32` view of both.
    trait PeekFixture: Fixture {
        fn peek_value(&mut self) -> Option<i32>;
    }
    trait PeekNthFixture: PeekFixture {
        fn peek_nth_value(&mut self, n: usize) -> Option<i32>;
    }
    trait PeekBackFixture: PeekFixture {
        fn peek_back_value(&mut self) -> Option<i32>;
    }
    trait PeekNthBackFixture: PeekBackFixture + PeekNthFixture {
        fn peek_nth_back_value(&mut self, n: usize) -> Option<i32>;
    }
    // Writes through the peeked reference, so only types that peek by `&mut` have it.
    trait PeekMutFixture: PeekFixture {
        fn poke(&mut self, value: i32);
    }
    trait PeekMutNthFixture: PeekMutFixture + PeekNthFixture {
        fn poke_nth_value(&mut self, n: usize, value: i32) -> Option<i32>;
    }
    trait PeekMutBackFixture: PeekMutFixture + PeekBackFixture {
        fn poke_back(&mut self, value: i32);
    }
    trait PeekMutNthBackFixture: PeekMutBackFixture + PeekMutNthFixture + PeekNthBackFixture {
        fn poke_nth_back_value(&mut self, n: usize, value: i32) -> Option<i32>;
    }

    trait HintedFixture: Iterator<Item = u8> {
        fn hinted() -> Self;
        fn peek_once(&mut self);
    }

    macro_rules! fixture {
        (peek, $ty:ty, $conv:expr) => {
            impl PeekFixture for $ty {
                fn peek_value(&mut self) -> Option<i32> {
                    self.peek().map($conv)
                }
            }
        };
        (peek_nth, $ty:ty, $conv:expr) => {
            impl PeekNthFixture for $ty {
                fn peek_nth_value(&mut self, n: usize) -> Option<i32> {
                    self.peek_nth(n).map($conv)
                }
            }
        };
        (peek_back, $ty:ty, $conv:expr) => {
            impl PeekBackFixture for $ty {
                fn peek_back_value(&mut self) -> Option<i32> {
                    self.peek_back().map($conv)
                }
            }
        };
        (peek_nth_back, $ty:ty, $conv:expr) => {
            impl PeekNthBackFixture for $ty {
                fn peek_nth_back_value(&mut self, n: usize) -> Option<i32> {
                    self.peek_nth_back(n).map($conv)
                }
            }
        };
        (poke, $ty:ty, $conv:expr) => {
            impl PeekMutFixture for $ty {
                fn poke(&mut self, value: i32) {
                    *self.peek().unwrap() = value;
                }
            }
        };
        (poke_nth, $ty:ty, $conv:expr) => {
            impl PeekMutNthFixture for $ty {
                fn poke_nth_value(&mut self, n: usize, value: i32) -> Option<i32> {
                    self.peek_nth(n).map(|x| std::mem::replace(x, value))
                }
            }
        };
        (poke_back, $ty:ty, $conv:expr) => {
            impl PeekMutBackFixture for $ty {
                fn poke_back(&mut self, value: i32) {
                    *self.peek_back().unwrap() = value;
                }
            }
        };
        (poke_nth_back, $ty:ty, $conv:expr) => {
            impl PeekMutNthBackFixture for $ty {
                fn poke_nth_back_value(&mut self, n: usize, value: i32) -> Option<i32> {
                    self.peek_nth_back(n).map(|x| std::mem::replace(x, value))
                }
            }
        };
    }

    macro_rules! fixtures {
        ($ty:ty, $conv:expr; $($kind:ident),+ $(,)?) => {
            $( fixture!($kind, $ty, $conv); )+
        };
    }

    macro_rules! hinted_fixture {
        ($ty:ty, $make:ident) => {
            impl HintedFixture for $ty {
                fn hinted() -> Self {
                    Hinted.$make()
                }

                fn peek_once(&mut self) {
                    self.peek();
                }
            }
        };
    }

    fixtures!(CopySpan, |x| x; peek, peek_nth);
    fixtures!(CloneRange, |x| x; peek, peek_nth);
    fixtures!(BufferRange, |x| *x; peek, peek_nth, poke, poke_nth);
    fixtures!(OptionRange, |x| *x; peek, poke);
    fixtures!(BackOptionRange, |x| *x; peek, peek_back, poke, poke_back);
    fixtures!(
        BackBufferRange, |x| *x;
        peek, peek_nth, peek_back, peek_nth_back, poke, poke_nth, poke_back, poke_nth_back,
    );

    type BufferHinted = PeekableWithBuffer<Hinted>;
    type OptionHinted = PeekableWithOption<Hinted>;
    type BackOptionHinted = PeekableBackWithOption<Hinted>;
    type BackBufferHinted = PeekableBackWithBuffer<Hinted>;

    hinted_fixture!(BufferHinted, peekable_with_buffer);
    hinted_fixture!(OptionHinted, peekable_with_option);
    hinted_fixture!(BackOptionHinted, peekable_back_with_option);
    hinted_fixture!(BackBufferHinted, peekable_back_with_buffer);

    #[generic_tests::define]
    mod generic {
        use super::*;

        #[test]
        fn priming_does_not_change_the_sequence<T: Fixture>() {
            each::<T>(|mk| {
                assert_eq!(mk(0..6).collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5]);
                assert_eq!(mk(0..6).rev().collect::<Vec<_>>(), [5, 4, 3, 2, 1, 0]);
                assert_eq!(mk(0..0).next(), None);
            });
        }

        #[test]
        fn next_size_hint_count_last<T: Fixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..5);
                assert_eq!(p.size_hint(), (5, Some(5)));
                assert_eq!(p.next(), Some(0));
                assert_eq!(p.size_hint(), (4, Some(4)));
                assert_eq!(p.count(), 4);

                assert_eq!(mk(0..4).count(), 4);
                assert_eq!(mk(0..4).last(), Some(3));
                assert_eq!(mk(0..0).count(), 0);
                assert_eq!(mk(0..0).last(), None);
            });
        }

        #[test]
        fn nth_skips_across_every_boundary<T: Fixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..9);

                assert_eq!(p.nth(0), Some(0));
                assert_eq!(p.nth(1), Some(2));
                assert_eq!(p.nth(0), Some(3));
                assert_eq!(p.nth(2), Some(6));
                assert_eq!(p.next(), Some(7));
                assert_eq!(p.nth(5), None);
                assert_eq!(p.next(), None);
            });
        }

        #[test]
        fn nth_at_and_past_the_end<T: Fixture>() {
            each::<T>(|mk| {
                assert_eq!(mk(0..6).nth(5), Some(5));
                assert_eq!(mk(0..6).nth(6), None);

                let mut p = mk(0..3);
                assert_eq!(p.nth(10), None);
                assert_eq!(p.next(), None);
            });
        }

        #[test]
        fn nth_past_end_empties_buffers<T: Fixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..8);

                assert_eq!(p.nth(20), None);
                assert_eq!(p.buffered(), 0);
            });
        }

        #[test]
        fn fold_for_each_reduce<T: Fixture>() {
            each::<T>(|mk| {
                assert_eq!(mk(1..5).fold(0, |acc, x| acc * 10 + x), 1234);

                let mut seen = Vec::new();
                mk(0..4).for_each(|x| seen.push(x));
                assert_eq!(seen, [0, 1, 2, 3]);

                assert_eq!(mk(1..5).reduce(|a, b| a * b), Some(24));
                assert_eq!(mk(0..0).reduce(|a, b| a + b), None);
            });
        }

        #[test]
        fn short_circuiting_results<T: Fixture>() {
            each::<T>(|mk| {
                assert!(mk(0..4).all(|x| x < 4));
                assert!(!mk(0..4).all(|x| x < 1));
                assert!(!mk(0..4).all(|x| x < 3));
                assert!(mk(0..4).any(|x| x == 0));
                assert!(mk(0..4).any(|x| x == 3));
                assert!(!mk(0..4).any(|x| x == 9));
                assert_eq!(mk(0..4).find(|x| x % 2 == 1), Some(1));
                assert_eq!(mk(0..4).find(|x| *x == 0), Some(0));
                assert_eq!(mk(0..4).find(|x| *x == 3), Some(3));
                assert_eq!(mk(0..4).find(|x| *x == 9), None);
                assert_eq!(mk(0..4).find_map(|x| (x > 2).then_some(x * 10)), Some(30));
                assert_eq!(mk(0..4).find_map(|x| (x > 9).then_some(x)), None);
                assert_eq!(mk(5..9).position(|x| x == 5), Some(0));
                assert_eq!(mk(5..9).position(|x| x == 8), Some(3));
                assert_eq!(mk(5..9).position(|x| x == 0), None);
            });
        }

        #[test]
        fn short_circuiting_leaves_the_rest<T: Fixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..6);
                assert!(p.any(|x| x == 1));
                assert_eq!(p.next(), Some(2));

                let mut p = mk(0..6);
                assert!(p.any(|x| x == 4));
                assert_eq!(p.next(), Some(5));

                let mut p = mk(0..6);
                assert!(!p.all(|x| x < 1));
                assert_eq!(p.next(), Some(2));

                let mut p = mk(0..6);
                assert!(!p.all(|x| x < 4));
                assert_eq!(p.next(), Some(5));

                let mut p = mk(0..6);
                assert_eq!(p.find(|x| *x == 1), Some(1));
                assert_eq!(p.next(), Some(2));

                let mut p = mk(0..6);
                assert_eq!(p.find(|x| *x == 4), Some(4));
                assert_eq!(p.next(), Some(5));

                let mut p = mk(0..6);
                assert_eq!(p.find_map(|x| (x == 1).then_some(x * 10)), Some(10));
                assert_eq!(p.next(), Some(2));

                let mut p = mk(0..6);
                assert_eq!(p.find_map(|x| (x == 4).then_some(x * 10)), Some(40));
                assert_eq!(p.next(), Some(5));

                let mut p = mk(0..6);
                assert_eq!(p.position(|x| x == 1), Some(1));
                assert_eq!(p.next(), Some(2));

                let mut p = mk(0..6);
                assert_eq!(p.position(|x| x == 4), Some(4));
                assert_eq!(p.next(), Some(5));

                let mut p = mk(0..6);
                assert_eq!(p.position(|x| x == 9), None);
                assert_eq!(p.next(), None);
            });
        }

        #[test]
        fn min_max_family<T: Fixture>() {
            each::<T>(|mk| {
                assert_eq!(mk(0..4).max(), Some(3));
                assert_eq!(mk(0..4).min(), Some(0));
                assert_eq!(mk(0..4).max_by(|a, b| b.cmp(a)), Some(0));
                assert_eq!(mk(0..4).min_by(|a, b| b.cmp(a)), Some(3));
                assert_eq!(mk(-2..2).max_by_key(|x| x.abs()), Some(-2));
                assert_eq!(mk(-2..2).min_by_key(|x| x.abs()), Some(0));
            });
        }

        #[test]
        fn sum_and_product<T: Fixture>() {
            each::<T>(|mk| {
                assert_eq!(mk(1..5).sum::<i32>(), 10);
                assert_eq!(mk(1..5).product::<i32>(), 24);
            });
        }

        #[test]
        fn partition_splits_by_predicate<T: Fixture>() {
            each::<T>(|mk| {
                let (even, odd): (Vec<_>, Vec<_>) = mk(0..6).partition(|x| x % 2 == 0);

                assert_eq!(even, [0, 2, 4]);
                assert_eq!(odd, [1, 3, 5]);
            });
        }

        #[test]
        fn comparisons<T: Fixture>() {
            each::<T>(|mk| {
                assert_eq!(mk(0..4).cmp(0..5), Ordering::Less);
                assert_eq!(mk(0..4).partial_cmp(0..4), Some(Ordering::Equal));
                assert!(mk(0..4).eq(0..4));
                assert!(mk(0..4).ne(0..5));
                assert!(mk(0..4).lt(0..5));
                assert!(mk(0..4).le(0..4));
                assert!(mk(0..4).gt(0..3));
                assert!(mk(0..4).ge(0..4));
            });
        }

        #[test]
        fn is_sorted_family<T: Fixture>() {
            each::<T>(|mk| {
                assert!(mk(0..4).is_sorted());
                assert!(mk(0..4).is_sorted_by(|a, b| a <= b));
                assert!(!mk(0..4).is_sorted_by(|a, b| a >= b));
                assert!(mk(0..4).is_sorted_by_key(|x| x));
                assert!(!mk(0..4).is_sorted_by_key(|x| -x));
            });
        }

        #[test]
        fn next_back_walks_the_whole_sequence<T: Fixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..5);

                for expected in (0..5).rev() {
                    assert_eq!(p.next_back(), Some(expected));
                }
                assert_eq!(p.next_back(), None);
                assert_eq!(p.next(), None);
            });
        }

        #[test]
        fn both_ends_meet_in_the_middle<T: Fixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..5);

                assert_eq!(p.next(), Some(0));
                assert_eq!(p.next_back(), Some(4));
                assert_eq!(p.next(), Some(1));
                assert_eq!(p.next_back(), Some(3));
                assert_eq!(p.next(), Some(2));
                assert_eq!(p.next_back(), None);
                assert_eq!(p.next(), None);
            });
        }

        #[test]
        fn nth_back_crosses_every_boundary<T: Fixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..5);

                assert_eq!(p.nth_back(3), Some(1));
                assert_eq!(p.next_back(), Some(0));
                assert_eq!(p.nth_back(0), None);
            });
        }

        #[test]
        fn rfold_visits_back_to_front<T: Fixture>() {
            each::<T>(|mk| {
                let reversed = mk(0..4).rfold(Vec::new(), |mut v, x| {
                    v.push(x);
                    v
                });

                assert_eq!(reversed, [3, 2, 1, 0]);
            });
        }

        #[test]
        fn rfind_results_and_leftovers<T: Fixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..6);
                assert_eq!(p.rfind(|x| *x == 4), Some(4));
                assert_eq!(p.next_back(), Some(3));

                let mut p = mk(0..6);
                assert_eq!(p.rfind(|x| *x == 1), Some(1));
                assert_eq!(p.next_back(), Some(0));

                let mut p = mk(0..6);
                assert_eq!(p.rfind(|x| *x == 0), Some(0));
                assert_eq!(p.next_back(), None);

                let mut p = mk(0..6);
                assert_eq!(p.rfind(|x| *x == 9), None);
                assert_eq!(p.next(), None);
            });
        }

        #[test]
        fn exact_size_tracks_both_ends<T: Fixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..5);

                assert_eq!(p.len(), 5);
                p.next();
                assert_eq!(p.len(), 4);
                p.next_back();
                assert_eq!(p.len(), 3);
            });
        }

        #[test]
        fn stays_exhausted<T: Fixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..2);

                p.by_ref().for_each(drop);

                assert_eq!(p.next(), None);
                assert_eq!(p.next(), None);
                assert_eq!(p.next_back(), None);
            });
        }

        #[test]
        fn clone_is_independent<T: Fixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..4);
                let q = p.clone();

                p.next();

                assert_eq!(q.collect::<Vec<_>>(), [0, 1, 2, 3]);
                assert_eq!(p.collect::<Vec<_>>(), [1, 2, 3]);
            });
        }

        #[instantiate_tests(<CopySpan>)]
        mod copy {}

        #[instantiate_tests(<CloneRange>)]
        mod clone {}

        #[instantiate_tests(<BufferRange>)]
        mod buffer {}

        #[instantiate_tests(<OptionRange>)]
        mod option {}

        #[instantiate_tests(<BackOptionRange>)]
        mod back_option {}

        #[instantiate_tests(<BackBufferRange>)]
        mod back_buffer {}
    }

    // PeekFixture
    #[generic_tests::define]
    mod generic_peek {
        use super::*;

        #[test]
        fn peek_does_not_consume<T: PeekFixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..3);

                assert_eq!(p.peek_value(), Some(0));
                assert_eq!(p.peek_value(), Some(0));
                assert_eq!(p.next(), Some(0));
                assert_eq!(p.peek_value(), Some(1));
                assert_eq!(p.collect::<Vec<_>>(), [1, 2]);
            });
        }

        #[test]
        fn peek_on_empty_is_none<T: PeekFixture>() {
            each::<T>(|mk| assert_eq!(mk(0..0).peek_value(), None));
        }

        #[instantiate_tests(<CopySpan>)]
        mod copy {}

        #[instantiate_tests(<CloneRange>)]
        mod clone {}

        #[instantiate_tests(<BufferRange>)]
        mod buffer {}

        #[instantiate_tests(<OptionRange>)]
        mod option {}

        #[instantiate_tests(<BackOptionRange>)]
        mod back_option {}

        #[instantiate_tests(<BackBufferRange>)]
        mod back_buffer {}
    }

    // PeekNthFixture
    #[generic_tests::define]
    mod generic_peek_nth {
        use super::*;

        #[test]
        fn peek_nth_does_not_consume<T: PeekNthFixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..5);

                assert_eq!(p.peek_nth_value(2), Some(2));
                assert_eq!(p.peek_nth_value(1), Some(1));
                assert_eq!(p.peek_nth_value(3), Some(3));
                assert_eq!(p.peek_nth_value(5), None);
                assert_eq!(p.collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
            });
        }

        #[test]
        fn peek_nth_on_empty_is_none<T: PeekNthFixture>() {
            each::<T>(|mk| assert_eq!(mk(0..0).peek_nth_value(3), None));
        }

        #[instantiate_tests(<CopySpan>)]
        mod copy {}

        #[instantiate_tests(<CloneRange>)]
        mod clone {}

        #[instantiate_tests(<BufferRange>)]
        mod buffer {}

        #[instantiate_tests(<BackBufferRange>)]
        mod back_buffer {}
    }

    // PeekBackFixture
    #[generic_tests::define]
    mod generic_peek_back {
        use super::*;

        #[test]
        fn peek_back_does_not_consume<T: PeekBackFixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..3);

                assert_eq!(p.peek_back_value(), Some(2));
                assert_eq!(p.peek_back_value(), Some(2));
                assert_eq!(p.next_back(), Some(2));
                assert_eq!(p.peek_back_value(), Some(1));
                assert_eq!(p.collect::<Vec<_>>(), [0, 1]);
            });
        }

        #[test]
        fn peek_back_on_empty_is_none<T: PeekBackFixture>() {
            each::<T>(|mk| assert_eq!(mk(0..0).peek_back_value(), None));
        }

        // The ends share one element here, so it has to move between the front and back stores.
        #[test]
        fn last_element_is_peekable_from_both_ends<T: PeekBackFixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..1);

                assert_eq!(p.peek_value(), Some(0));
                assert_eq!(p.peek_back_value(), Some(0));
                assert_eq!(p.peek_value(), Some(0));
                assert_eq!(p.collect::<Vec<_>>(), [0]);
            });
        }

        #[test]
        fn peek_both_ends_of_two_items<T: PeekBackFixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..2);

                assert_eq!(p.peek_value(), Some(0));
                assert_eq!(p.peek_back_value(), Some(1));
                assert_eq!(p.collect::<Vec<_>>(), [0, 1]);
            });
        }

        #[instantiate_tests(<BackOptionRange>)]
        mod back_option {}

        #[instantiate_tests(<BackBufferRange>)]
        mod back_buffer {}
    }

    // PeekNthBackFixture
    #[generic_tests::define]
    mod generic_peek_nth_back {
        use super::*;

        #[test]
        fn peek_nth_back_does_not_consume<T: PeekNthBackFixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..5);

                assert_eq!(p.peek_nth_back_value(2), Some(2));
                assert_eq!(p.peek_nth_back_value(0), Some(4));
                assert_eq!(p.peek_nth_back_value(3), Some(1));
                assert_eq!(p.peek_nth_back_value(5), None);
                assert_eq!(p.collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
            });
        }

        #[test]
        fn peek_nth_back_on_empty_is_none<T: PeekNthBackFixture>() {
            each::<T>(|mk| assert_eq!(mk(0..0).peek_nth_back_value(3), None));
        }

        #[instantiate_tests(<BackBufferRange>)]
        mod back_buffer {}
    }

    // PeekMutFixture
    #[generic_tests::define]
    mod generic_peek_mut {
        use super::*;

        #[test]
        fn peek_allows_mutation<T: PeekMutFixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..3);

                p.poke(10);

                assert_eq!(p.collect::<Vec<_>>(), [10, 1, 2]);
            });
        }

        #[instantiate_tests(<BufferRange>)]
        mod buffer {}

        #[instantiate_tests(<OptionRange>)]
        mod option {}

        #[instantiate_tests(<BackOptionRange>)]
        mod back_option {}

        #[instantiate_tests(<BackBufferRange>)]
        mod back_buffer {}
    }

    // PeekMutNthFixture
    #[generic_tests::define]
    mod generic_peek_mut_nth {
        use super::*;

        #[test]
        fn poke_nth_returns_old_value_and_writes<T: PeekMutNthFixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..4);

                assert_eq!(p.poke_nth_value(1, 11), Some(1));
                assert_eq!(p.poke_nth_value(1, 12), Some(11));
                assert_eq!(p.poke_nth_value(4, 0), None);
                assert_eq!(p.collect::<Vec<_>>(), [0, 12, 2, 3]);
            });
        }

        #[instantiate_tests(<BufferRange>)]
        mod buffer {}

        #[instantiate_tests(<BackBufferRange>)]
        mod back_buffer {}
    }

    // PeekMutBackFixture
    #[generic_tests::define]
    mod generic_peek_mut_back {
        use super::*;

        #[test]
        fn peek_allows_mutation_at_both_ends<T: PeekMutBackFixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..3);

                p.poke(10);
                p.poke_back(20);

                assert_eq!(p.collect::<Vec<_>>(), [10, 1, 20]);
            });
        }

        #[test]
        fn mutation_survives_moving_between_ends<T: PeekMutBackFixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..1);

                p.poke(10);

                assert_eq!(p.peek_back_value(), Some(10));
                assert_eq!(p.collect::<Vec<_>>(), [10]);
            });
        }

        #[instantiate_tests(<BackOptionRange>)]
        mod back_option {}

        #[instantiate_tests(<BackBufferRange>)]
        mod back_buffer {}
    }

    // PeekMutNthBackFixture
    #[generic_tests::define]
    mod generic_peek_mut_nth_back {
        use super::*;

        #[test]
        fn poke_nth_back_returns_old_value_and_writes<T: PeekMutNthBackFixture>() {
            each::<T>(|mk| {
                let mut p = mk(0..4);

                assert_eq!(p.poke_nth_back_value(1, 21), Some(2));
                assert_eq!(p.poke_nth_back_value(1, 22), Some(21));
                assert_eq!(p.poke_nth_back_value(4, 0), None);
                assert_eq!(p.collect::<Vec<_>>(), [0, 1, 22, 3]);
            });
        }

        #[instantiate_tests(<BackBufferRange>)]
        mod back_buffer {}
    }

    // HintedFixture
    #[generic_tests::define]
    mod generic_hinted {
        use super::*;

        #[test]
        fn size_hint_saturates<T: HintedFixture>() {
            let mut p = T::hinted();

            assert_eq!(p.size_hint(), (usize::MAX, Some(usize::MAX)));
            p.peek_once();
            assert_eq!(p.size_hint(), (usize::MAX, None));
        }

        #[instantiate_tests(<BufferHinted>)]
        mod buffer {}

        #[instantiate_tests(<OptionHinted>)]
        mod option {}

        #[instantiate_tests(<BackOptionHinted>)]
        mod back_option {}

        #[instantiate_tests(<BackBufferHinted>)]
        mod back_buffer {}
    }

    #[derive(Debug, Copy)]
    struct HasLoudClone {
        value: i32,
        clone_counter: usize,
    }

    impl HasLoudClone {
        fn new(value: i32) -> Self {
            Self {
                value,
                clone_counter: 0,
            }
        }
    }

    impl Clone for HasLoudClone {
        fn clone(&self) -> Self {
            Self {
                value: self.value,
                clone_counter: self.clone_counter + 1,
            }
        }
    }

    impl std::cmp::PartialEq for HasLoudClone {
        fn eq(&self, other: &Self) -> bool {
            self.partial_cmp(other).is_some_and(|cmp| cmp.is_eq())
        }
    }
    impl std::cmp::PartialOrd for HasLoudClone {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.value.partial_cmp(&other.value)
        }
    }
    impl std::ops::Add<i32> for HasLoudClone {
        type Output = HasLoudClone;

        fn add(mut self, rhs: i32) -> Self::Output {
            self.value += rhs;
            self
        }
    }

    impl std::borrow::Borrow<i32> for HasLoudClone {
        fn borrow(&self) -> &i32 {
            &self.value
        }
    }

    mod peekable_with_copy {
        use std::assert_matches;

        use super::*;

        #[test]
        fn copies_without_cloning() {
            let mut p = Span::from(HasLoudClone::new(0)..HasLoudClone::new(3)).peekable_with_copy();
            let mut q = p.clone();

            q.next();

            assert_eq!(p.clone().count(), 3);
            assert_eq!(q.clone().count(), 2);

            p.peek();

            assert_matches!(
                p.next(),
                Some(HasLoudClone {
                    value: 0,
                    clone_counter: 0
                })
            );
            assert_matches!(
                q.next(),
                Some(HasLoudClone {
                    value: 1,
                    clone_counter: 0
                })
            );
        }
    }

    mod peekable_with_clone {
        use std::assert_matches;

        use super::*;

        #[test]
        fn peeks_non_copy_items_by_cloning() {
            let mut p = vec!["a".to_string(), "b".to_string()]
                .into_iter()
                .peekable_with_clone();

            assert_eq!(p.peek(), Some("a".to_string()));
            assert_eq!(p.peek_nth(1), Some("b".to_string()));
            assert_eq!(p.next(), Some("a".to_string()));

            let mut p =
                Span::from(HasLoudClone::new(0)..HasLoudClone::new(3)).peekable_with_clone();

            assert_matches!(
                p.next(),
                Some(HasLoudClone {
                    value: 0,
                    clone_counter: 0
                })
            );
            assert_matches!(
                p.peek(),
                Some(HasLoudClone {
                    value: 1,
                    clone_counter: 1
                })
            );
            assert_matches!(
                p.peek_nth(1),
                Some(HasLoudClone {
                    value: 2,
                    clone_counter: 1
                })
            );
            // The original iterator remains uncloned
            assert_matches!(
                p.next(),
                Some(HasLoudClone {
                    value: 1,
                    clone_counter: 0
                })
            );
        }
    }

    mod peekable_with_buffer {
        use super::*;

        fn buf(start: i32, end: i32, peeked: usize) -> PeekableWithBuffer<Range<i32>> {
            PeekableWithBuffer::primed(start..end, pr(peeked, 0))
        }

        #[test]
        fn peek_buffers_one_item() {
            let mut p = buf(0, 3, 0);

            p.peek();
            p.peek();

            assert_eq!(p.buffer, [0]);
        }

        #[test]
        fn peek_nth_buffers_up_to_n() {
            let mut p = buf(0, 5, 0);

            assert_eq!(p.peek_nth(2), Some(&mut 2));
            assert_eq!(p.buffer, [0, 1, 2]);
            assert_eq!(p.peek_nth(1), Some(&mut 1));
            assert_eq!(p.buffer, [0, 1, 2]);
            assert_eq!(p.peek_nth(3), Some(&mut 3));
            assert_eq!(p.buffer, [0, 1, 2, 3]);
            assert_eq!(p.collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
        }

        #[test]
        fn peek_nth_past_end_keeps_items() {
            let mut p = buf(0, 3, 0);

            assert_eq!(p.peek_nth(3), None);
            assert_eq!(p.buffer, [0, 1, 2]);
            assert_eq!(p.collect::<Vec<_>>(), [0, 1, 2]);
        }

        #[test]
        #[should_panic(expected = "assertion failed: n < usize::MAX")]
        fn peek_nth_usize_max_panics() {
            buf(0, 3, 0).peek_nth(usize::MAX);
        }

        #[test]
        fn peek_nth_pulls_only_what_it_needs() {
            let mut p = Counter { count: 0 }.peekable_with_buffer();

            p.peek_nth(4);
            assert_eq!(p.inner.count, 5);

            p.next();
            p.next();
            assert_eq!(p.inner.count - p.buffer.len(), 2);

            p.peek_nth(2);
            assert_eq!(p.inner.count, 5);
        }

        #[test]
        fn nth_within_buffer_keeps_the_rest_buffered() {
            let mut p = buf(0, 6, 4);

            assert_eq!(p.nth(1), Some(1));
            assert_eq!(p.buffer, [2, 3]);
        }

        #[test]
        fn nth_past_buffer_empties_it() {
            let mut p = buf(0, 8, 3);

            assert_eq!(p.nth(3), Some(3));
            assert!(p.buffer.is_empty());
        }

        #[test]
        fn pop_buffer_pops_from_both_ends() {
            let mut deque: VecDeque<i32> = (0..4).collect();
            let mut pop = PopBuffer(&mut deque);

            assert_eq!(pop.size_hint(), (4, Some(4)));
            assert_eq!(pop.len(), 4);
            assert_eq!(pop.next(), Some(0));
            assert_eq!(pop.next_back(), Some(3));
            assert_eq!(pop.len(), 2);
            assert_fused(&pop);
            assert_eq!(deque, [1, 2]);
        }
    }

    mod peekable_with_option {
        use super::*;

        fn opt(start: i32, end: i32, peeked: bool) -> PeekableWithOption<Range<i32>> {
            PeekableWithOption::primed(start..end, pr(peeked.into(), 0))
        }

        #[test]
        fn peek_fills_slot() {
            let mut p = opt(0, 3, false);

            p.peek();
            assert_eq!(p.peeked, Some(0));
            p.next();
            assert_eq!(p.peeked, None);
        }

        #[test]
        fn peek_pulls_at_most_one() {
            let mut p = Counter { count: 0 }.peekable_with_option();

            p.peek();
            p.peek();
            assert_eq!(p.inner.count, 1);

            p.next();
            assert_eq!(p.inner.count, 1);
            p.next();
            assert_eq!(p.inner.count, 2);
        }

        #[test]
        fn peek_on_exhausted_polls_again() {
            let mut p = Flaky(vec![Some(1), None, Some(2)].into_iter()).peekable_with_option();

            assert_eq!(p.peek(), Some(&mut 1));
            assert_eq!(p.next(), Some(1));
            assert_eq!(p.peek(), None);
            assert_eq!(p.peek(), Some(&mut 2));
        }

        #[test]
        fn take_option_yields_once() {
            let mut slot = Some(1);
            let mut take = TakeOption(&mut slot);

            assert_eq!(take.size_hint(), (1, Some(1)));
            assert_eq!(take.len(), 1);
            assert_eq!(take.next(), Some(1));
            assert_eq!(take.size_hint(), (0, Some(0)));
            assert_eq!(take.next(), None);
            assert_fused(&take);
            assert_eq!(slot, None);
        }

        #[test]
        fn take_option_next_back() {
            let mut slot = Some(2);
            let mut take = TakeOption(&mut slot);

            assert_eq!(take.next_back(), Some(2));
            assert_eq!(take.next_back(), None);
            assert_eq!(slot, None);
        }
    }

    mod peekable_back_with_option {
        use super::*;

        fn opt(start: i32, end: i32, priming: Priming) -> PeekableBackWithOption<Range<i32>> {
            PeekableBackWithOption::primed(start..end, priming)
        }

        #[test]
        fn peek_fills_each_slot() {
            let mut p = opt(0, 3, pr(0, 0));

            p.peek();
            assert_eq!((p.front, p.back), (Some(0), None));
            p.peek_back();
            assert_eq!((p.front, p.back), (Some(0), Some(2)));
            p.next();
            p.next_back();
            assert_eq!((p.front, p.back), (None, None));
        }

        #[test]
        fn peek_pulls_one_from_each_end() {
            let mut p = opt(0, 5, pr(0, 0));

            p.peek();
            p.peek_back();

            assert_eq!((p.front, p.back), (Some(0), Some(4)));
            assert_eq!(p.inner, 1..4);
        }

        #[test]
        fn last_element_moves_between_slots() {
            let mut p = opt(0, 1, pr(0, 0));

            assert_eq!(p.peek(), Some(&mut 0));
            assert_eq!((p.front, p.back), (Some(0), None));

            assert_eq!(p.peek_back(), Some(&mut 0));
            assert_eq!((p.front, p.back), (None, Some(0)));

            assert_eq!(p.peek(), Some(&mut 0));
            assert_eq!((p.front, p.back), (Some(0), None));

            assert_eq!(p.collect::<Vec<_>>(), [0]);
        }

        #[test]
        fn both_slots_filled_and_inner_drained() {
            let mut p = opt(0, 2, pr(0, 0));

            p.peek();
            p.peek_back();

            assert_eq!((p.front, p.back), (Some(0), Some(1)));
            assert_eq!(p.inner, 1..1);
        }

        #[test]
        fn nth_skips_front_and_leaves_back() {
            // front = 0, inner = 1..4, back = 4
            let mut p = opt(0, 5, pr(1, 1));

            assert_eq!(p.nth(2), Some(2));
            assert_eq!((p.front, p.back), (None, Some(4)));
            assert_eq!(p.inner, 3..4);
        }

        #[test]
        fn nth_lands_in_back_slot() {
            let mut p = opt(0, 5, pr(1, 1));

            assert_eq!(p.nth(4), Some(4));
            assert_eq!((p.front, p.back), (None, None));
        }

        #[test]
        fn nth_with_empty_front_uses_inner() {
            let mut p = opt(0, 5, pr(0, 1));

            assert_eq!(p.nth(1), Some(1));
            assert_eq!(p.back, Some(4));
        }
    }

    mod peekable_back_with_buffer {
        use super::*;

        fn buf(start: i32, end: i32, priming: Priming) -> PeekableBackWithBuffer<Range<i32>> {
            PeekableBackWithBuffer::primed(start..end, priming)
        }

        #[test]
        fn peek_fills_each_buffer() {
            let mut p = buf(0, 3, pr(0, 0));

            p.peek();
            assert_eq!((p.front.len(), p.back.len()), (1, 0));
            p.peek_back();
            assert_eq!((p.front.len(), p.back.len()), (1, 1));
        }

        #[test]
        fn peek_nth_buffers_up_to_n() {
            let mut p = buf(0, 5, pr(0, 0));

            assert_eq!(p.peek_nth(2), Some(&mut 2));
            assert_eq!(p.front, [0, 1, 2]);
            assert_eq!(p.inner, 3..5);
            assert_eq!(p.peek_nth(1), Some(&mut 1));
            assert_eq!(p.front, [0, 1, 2]);
            assert_eq!(p.peek_nth(3), Some(&mut 3));
            assert_eq!(p.front, [0, 1, 2, 3]);
            assert_eq!(p.collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
        }

        #[test]
        fn peek_nth_back_buffers_in_sequence_order() {
            let mut p = buf(0, 5, pr(0, 0));

            assert_eq!(p.peek_nth_back(2), Some(&mut 2));
            assert_eq!(p.back, [2, 3, 4]);
            assert_eq!(p.inner, 0..2);
            assert_eq!(p.peek_nth_back(0), Some(&mut 4));
            assert_eq!(p.peek_nth_back(1), Some(&mut 3));
            assert_eq!(p.back, [2, 3, 4]);
            assert_eq!(p.peek_nth_back(3), Some(&mut 1));
            assert_eq!(p.back, [1, 2, 3, 4]);
            assert_eq!(p.collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
        }

        #[test]
        fn peek_nth_past_end_keeps_items() {
            let mut p = buf(0, 3, pr(0, 0));

            assert_eq!(p.peek_nth(3), None);
            assert_eq!(p.front, [0, 1, 2]);
            assert_eq!(p.collect::<Vec<_>>(), [0, 1, 2]);
        }

        #[test]
        fn peek_nth_back_past_end_keeps_items() {
            let mut p = buf(0, 3, pr(0, 0));

            assert_eq!(p.peek_nth_back(3), None);
            assert_eq!(p.back, [0, 1, 2]);
            assert_eq!(p.collect::<Vec<_>>(), [0, 1, 2]);
        }

        #[test]
        #[should_panic(expected = "assertion failed: n < usize::MAX")]
        fn peek_nth_usize_max_panics() {
            buf(0, 3, pr(0, 0)).peek_nth(usize::MAX);
        }

        #[test]
        #[should_panic(expected = "assertion failed: n < usize::MAX")]
        fn peek_nth_back_usize_max_panics() {
            buf(0, 3, pr(0, 0)).peek_nth_back(usize::MAX);
        }

        #[test]
        fn peek_nth_takes_from_back_buffer_once_inner_is_dry() {
            let mut p = buf(0, 5, pr(0, 3));

            assert_eq!(p.peek_nth(3), Some(&mut 3));
            assert_eq!(p.front, [0, 1, 2, 3]);
            assert_eq!(p.back, [4]);
            assert_eq!(p.collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
        }

        #[test]
        fn peek_nth_back_takes_from_front_buffer_once_inner_is_dry() {
            let mut p = buf(0, 5, pr(3, 0));

            assert_eq!(p.peek_nth_back(3), Some(&mut 1));
            assert_eq!(p.front, [0]);
            assert_eq!(p.back, [1, 2, 3, 4]);
            assert_eq!(p.collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
        }

        #[test]
        fn peek_with_dry_inner_moves_an_item_across() {
            let mut p = buf(0, 2, pr(0, 2));

            assert_eq!(p.peek(), Some(&mut 0));
            assert_eq!(p.front, [0]);
            assert_eq!(p.back, [1]);
        }

        #[test]
        fn peek_back_with_dry_inner_moves_an_item_across() {
            let mut p = buf(0, 2, pr(2, 0));

            assert_eq!(p.peek_back(), Some(&mut 1));
            assert_eq!(p.front, [0]);
            assert_eq!(p.back, [1]);
        }

        #[test]
        fn nth_within_front_keeps_the_rest_buffered() {
            let mut p = buf(0, 6, pr(4, 0));

            assert_eq!(p.nth(1), Some(1));
            assert_eq!(p.front, [2, 3]);
        }

        #[test]
        fn nth_within_inner_leaves_back_buffer() {
            // front = [0, 1], inner = 2..5, back = [5, 6, 7]
            let mut p = buf(0, 8, pr(2, 3));

            assert_eq!(p.nth(3), Some(3));
            assert!(p.front.is_empty());
            assert_eq!(p.inner, 4..5);
            assert_eq!(p.back, [5, 6, 7]);
        }

        #[test]
        fn nth_lands_on_first_back_item() {
            let mut p = buf(0, 8, pr(2, 3));

            assert_eq!(p.nth(5), Some(5));
            assert_eq!(p.back, [6, 7]);
        }

        #[test]
        fn nth_skips_into_back_buffer() {
            let mut p = buf(0, 8, pr(2, 3));

            assert_eq!(p.nth(6), Some(6));
            assert_eq!(p.back, [7]);
        }
    }
}
