#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![deny(missing_docs)]
//! Provides several traits for peeking ahead of iterators. In particular, see [`Peek`],
//! [`PeekNth`], [`PeekBack`], and [`PeekNthBack`].

mod impls;
mod with;

pub use with::{
    MakePeekableBackWithBuffer, MakePeekableBackWithOption, MakePeekableWithBuffer,
    MakePeekableWithClone, MakePeekableWithCopy, MakePeekableWithOption, PeekableBackWithBuffer,
    PeekableBackWithOption, PeekableWithBuffer, PeekableWithClone, PeekableWithCopy,
    PeekableWithOption,
};

/// A generic trait to allow peeking into iterators.
pub trait Peek: Iterator {
    /// The item returned by `peek`. May borrow `self`. Is usually either `<Self as Iterator>::Item`
    /// or `&'a mut <Self as Iterator>::Item`.
    type PeekItem<'a>
    where
        Self: 'a;

    /// Peeks ahead at the next item in an iterator.
    fn peek<'a>(&'a mut self) -> Option<Self::PeekItem<'a>>;
    /// Consumes and returns the next item in the iterator if a condition is true.
    fn next_if<'a, P: FnOnce(Self::PeekItem<'_>) -> bool>(&mut self, p: P) -> Option<Self::Item> {
        if p(self.peek()?) { self.next() } else { None }
    }
}

/// A generic trait to allow peeking arbitrarily deep into iterators.
pub trait PeekNth: Peek {
    /// Peek ahead at the nth item in an iterator.
    fn peek_nth<'a>(&'a mut self, n: usize) -> Option<Self::PeekItem<'a>>;
}

/// A generic trait to allow peeking from the back of an iterator.
pub trait PeekBack: DoubleEndedIterator + Peek {
    /// Peeks at the back of a double-ended iterator.
    fn peek_back<'a>(&'a mut self) -> Option<Self::PeekItem<'a>>;
    /// Consumes and returns the next item from the back of the iterator if a condition is true.
    fn next_back_if<'a, P: FnOnce(Self::PeekItem<'_>) -> bool>(
        &mut self,
        p: P,
    ) -> Option<Self::Item> {
        if p(self.peek_back()?) {
            self.next_back()
        } else {
            None
        }
    }
}

/// A generic trait to allow peeking arbitrarily deep into the back of an iterator.
pub trait PeekNthBack: PeekBack + PeekNth {
    /// Peeks at the nth item from the back of an iterator.
    fn peek_nth_back<'a>(&'a mut self, n: usize) -> Option<Self::PeekItem<'a>>;
}
